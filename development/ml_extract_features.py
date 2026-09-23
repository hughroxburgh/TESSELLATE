"""
Extract tessellate.ml_classifier features for a sector. Run this on the
cluster, where the reduced flux cubes live (~1 GB per cut); the output table
is small enough to copy anywhere for training.

Edit the CONFIG block, then:  python ml_extract_features.py

With CACHE_DIR set, every finished cut is saved there, so if the job dies,
re-running the script carries on where it stopped.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import FEATURE_GROUPS, build_feature_table

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTOR = 55
CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]
CUTS = range(1, 65)                     # all 64 cuts

OUT_DIR = '/fred/oz335/hroxburg/dev/ml_classifier'
OUT_FILE = f'{OUT_DIR}/S{SECTOR}_features.csv.gz'
CACHE_DIR = f'{OUT_DIR}/S{SECTOR}_feature_cache'

EVENTS_CSV = None           # a csv of events (sector/camera/ccd/cut/objid/eventid) to restrict to, e.g.
                            # the sorted ones -- much faster than the whole sector. None = every event.
N_JOBS = 16                 # match the number of CPUs you request
CROSSMATCH = True           # Gaia / variable-catalogue features (needs the WCS + local catalogues)
FRAME_STATS = True          # per-frame noise pass over each cube (~10 s per cut)
OVERWRITE = False           # True = recompute cuts already in CACHE_DIR
# ----------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events = pd.read_csv(EVENTS_CSV) if EVENTS_CSV else None

    start = time.time()
    features = build_feature_table(DATA_PATH, SECTOR, cams=CAMS, ccds=CCDS, cuts=CUTS, events=events,
                                   cache_dir=CACHE_DIR, overwrite=OVERWRITE, n_jobs=N_JOBS,
                                   config={'crossmatch': CROSSMATCH, 'frame_stats': FRAME_STATS})
    features.to_csv(OUT_FILE, index=False)

    elapsed = time.time() - start
    print(f'\n{len(features)} events, {features.groupby(["camera", "ccd", "cut"]).ngroups} cuts '
          f'in {elapsed / 60:.1f} min -> {OUT_FILE}')
    print('Fraction of NaN per feature group:')
    for g in FEATURE_GROUPS:
        cols = [c for c in features if c.startswith(f'{g}_')]
        if cols:
            print(f'  {g:6s} {len(cols):3d} features  {np.mean(features[cols].isna().to_numpy()):.2f}')


if __name__ == '__main__':
    main()

"""
Draw a uniform random pool of events for the ML sort sample. Run on the cluster (reads detected_events.csv).

Two stages, so only a few hundred cuts need reading: CUTS_PER_SECTOR cuts are picked at random in each sector
(from those with a detected_events.csv), then POOL_SIZE events are drawn uniformly from all events in those cuts
in FRAME_BINS -- every event, pipeline-tagged or not, sorted or not, with no filter_events. Each event's chance
of being in the pool is saved as p_pool, so numbers from the sort can be reweighted to the whole sectors.

Next: extract the pool's features with ml_extract_features.py (LABELS_CSV = None, EVENTS_CSV = the pool csv,
TAGGED_PER_CUT = None so no tagged event is dropped, CROSSMATCH = False, CACHE_NAME = 'S{sector}_feature_cache_pool'),
collect them with ml_collect_features.py (RANDOM_SAMPLE_CSV = the pool csv, same CACHE_NAME), copy the table down,
then ml_pick_sort_sample.py (local) scores the pool and picks the events to sort.

Edit the CONFIG block, then:  python ml_random_pool.py
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tessellate.tools import load_table, table_exists

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTORS = list(range(27, 40)) + [55]
CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]
N_CUTS = 64                  # cuts per CCD
CUTS_PER_SECTOR = 30         # cuts drawn at random in each sector
POOL_SIZE = 20000            # events drawn from all of them
FRAME_BINS = [1]
SEED = 2026
N_JOBS = 4                   # files read in parallel (keep small on the login node)

OUT_FILE = '/fred/oz335/hroxburg/dev/ml_classifier/random_pool.csv'
# ----------------

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']


def cut_path(sector, cam, ccd, cut):
    return f'{DATA_PATH}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{N_CUTS}/detected_events.csv'


def read_cut(sector, cam, ccd, cut):
    ev = load_table(cut_path(sector, cam, ccd, cut), columns=['objid', 'eventid', 'frame_bin', 'classification'])
    ev = ev[ev.frame_bin.isin(FRAME_BINS)]
    return ev.assign(sector=sector, camera=cam, ccd=ccd, cut=cut)


def main():
    rng = np.random.default_rng(SEED)
    picked, frac = [], {}
    for sector in SECTORS:
        cuts = [(sector, cam, ccd, cut) for cam in CAMS for ccd in CCDS for cut in range(1, N_CUTS + 1)
                if table_exists(cut_path(sector, cam, ccd, cut))]
        if not cuts:
            print(f'Sector {sector}: no detected_events.csv found; skipped')
            continue
        k = min(CUTS_PER_SECTOR, len(cuts))
        chosen = [cuts[i] for i in rng.choice(len(cuts), k, replace=False)]
        picked += chosen
        frac[sector] = k / len(cuts)          # chance a given cut of this sector was read
        print(f'Sector {sector}: {k} of {len(cuts)} cuts', flush=True)

    parts = Parallel(n_jobs=N_JOBS)(delayed(read_cut)(*c) for c in picked)
    events = pd.concat(parts, ignore_index=True)
    n = min(POOL_SIZE, len(events))
    pool = events.iloc[np.sort(rng.choice(len(events), n, replace=False))].copy()
    pool['p_pool'] = pool['sector'].map(frac) * n / len(events)   # chance each event made it into the pool
    pool = pool[KEY_COLS + ['frame_bin', 'classification', 'p_pool']]

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    pool.to_csv(OUT_FILE, index=False)
    print(f'\n{len(events)} events in {len(picked)} cuts; drew {n} -> {OUT_FILE}')
    print('Pool by pipeline classification:')
    print(pool['classification'].value_counts().to_string())
    print('Pool by sector:')
    print(pool['sector'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()

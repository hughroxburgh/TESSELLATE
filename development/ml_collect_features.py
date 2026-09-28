"""
Build the classifier's training table from the per-cut feature files written
by ml_extract_features.py. A whole sector is tens of millions of events --
far too many to load -- so this keeps only the rows training and evaluation
need, with a column `kept_as` saying why:

  labelled    has an image in one of the sort folders (SORT_DIRS)
  sibling     the same physical event at another frame bin (shares a
              crossbin id with a labelled event), so it inherits the label
  random      in RANDOM_SAMPLE_CSV (sample_random_events.py), if given
  tagged      the pipeline's Junk / CosmicRay / Asteroid tags (already capped
              per cut by TAGGED_PER_CUT at extraction)
  background  a uniform random FRACTION of everything else -- for the review
              queue and a look at the population the classifier will meet

Run it on the cluster, then copy OUT_FILE down to development/S55/. It only
reads the feature files, so it can run while the extraction is still going
(files written in the last SKIP_RECENT_MIN minutes are left out, in case
they're still being written), and again at any time, e.g. once the extraction
has finished or the random sample has been sorted.

Edit the CONFIG block, then:  python ml_collect_features.py
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import (FEATURE_GROUPS, FEATURE_VERSION, KEY_COLS, PIPELINE_CLASSES, _parse_ids,
                                      load_manual_labels)

# ---- CONFIG ----
SECTOR = 55
OUT_DIR = '/fred/oz335/hroxburg/dev/ml_classifier'
CACHE_DIR = f'{OUT_DIR}/S{SECTOR}_feature_cache'     # the CACHE_DIR used by ml_extract_features.py
SORT_DIRS = ['/fred/oz335/hroxburg/dev/final_localisation/sort_found_flares',   # manual_sort outputs
             '/fred/oz335/hroxburg/dev/final_localisation/sort_non_flares']
RANDOM_SAMPLE_CSV = None    # random_sample.csv from sample_random_events.py, to keep those events (None = skip)
FRACTION = 0.01             # share of all other events kept as background (the same chance in every cut)
SEED = 0
N_JOBS = 8                  # files read in parallel (use 2-4 on the login node)
SKIP_RECENT_MIN = 3         # leave out files written in the last few minutes -- they may be half-written

OUT_FILE = f'{OUT_DIR}/S{SECTOR}_training_features.csv.gz'
# ----------------


def _keys(df):
    return set(map(tuple, df[KEY_COLS].astype(int).to_numpy()))


def collect_cut(path, labelled, random):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f'  {os.path.basename(path)}: skipped, unreadable ({type(e).__name__})')
        return None
    if not len(df):
        return None
    keys = list(map(tuple, df[KEY_COLS].astype(int).to_numpy()))
    is_lab = np.array([k in labelled for k in keys])
    is_rand = np.array([k in random for k in keys])

    ids = df['crossbin_ids'].apply(_parse_ids) if 'crossbin_ids' in df else pd.Series([[]] * len(df))
    lab_ids = {i for lst in ids[is_lab] for i in lst}
    is_sib = ~is_lab & ids.apply(lambda lst: any(i in lab_ids for i in lst)).to_numpy()
    is_tag = df['classification'].isin(PIPELINE_CLASSES).to_numpy()

    first = df.iloc[0]
    rng = np.random.default_rng([SEED, int(first.sector), int(first.camera), int(first.ccd), int(first.cut)])
    is_bg = rng.random(len(df)) < FRACTION

    kept_as = np.select([is_lab, is_sib, is_rand, is_tag, is_bg],
                        ['labelled', 'sibling', 'random', 'tagged', 'background'], '')
    keep = kept_as != ''
    return df[keep].assign(kept_as=kept_as[keep])


def main():
    from joblib import Parallel, delayed
    from tqdm import tqdm

    files = sorted(glob.glob(f'{CACHE_DIR}/S{SECTOR}C*_features_v{FEATURE_VERSION}.csv'))
    if not files:
        sys.exit(f'No S{SECTOR}C*_features_v{FEATURE_VERSION}.csv files in {CACHE_DIR}')
    recent = [f for f in files if time.time() - os.path.getmtime(f) < SKIP_RECENT_MIN * 60]
    files = [f for f in files if f not in recent]
    if recent:
        print(f'Leaving out {len(recent)} files written in the last {SKIP_RECENT_MIN} min (may be half-written)')
    labels = pd.concat([load_manual_labels(d) for d in SORT_DIRS], ignore_index=True)
    labelled = _keys(labels[labels.sector == SECTOR])
    random = _keys(pd.read_csv(RANDOM_SAMPLE_CSV)) if RANDOM_SAMPLE_CSV else set()
    print(f'{len(files)} feature files; {len(labelled)} labelled events; {len(random)} random-sample events')

    start = time.time()
    parts = Parallel(n_jobs=N_JOBS)(delayed(collect_cut)(f, labelled, random) for f in tqdm(files, ascii=True))
    table = pd.concat([p for p in parts if p is not None], ignore_index=True)
    table.to_csv(OUT_FILE, index=False)

    found = table.kept_as.eq('labelled').sum()
    print(f'\n{len(table)} rows in {(time.time() - start) / 60:.1f} min -> {OUT_FILE}')
    print('  ' + table.kept_as.value_counts().to_string().replace('\n', '\n  '))
    if found < len(labelled):
        print(f'  note: {len(labelled) - found} labelled events are not in the feature files '
              f'(cuts not extracted yet, or detection re-run since sorting)')
    if random and table.kept_as.eq('random').sum() < len(random):
        print(f'  note: {len(random) - table.kept_as.eq("random").sum()} random-sample events are not in the files')
    print('Fraction of NaN per feature group:')
    for g in FEATURE_GROUPS:
        cols = [c for c in table if c.startswith(f'{g}_')]
        if cols:
            print(f'  {g:6s} {len(cols):3d} features  {np.mean(table[cols].isna().to_numpy()):.2f}')


if __name__ == '__main__':
    main()

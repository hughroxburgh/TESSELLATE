"""
Pull the saved features of hand-labelled events out of a classified sector (ml_classify_sector.py wrote one
features file per cut), so a fresh sector's sorts can be added to training without re-extracting. Run on the
cluster, then copy OUT_FILE down to development/ml_data/v5/.

The events are the ones in the manual_sort folders in SORT_DIRS (every */events.csv, whatever the label) and
in any csv in EVENTS_CSVS (needs sector/camera/ccd/cut/objid/eventid). Only the cuts holding one of them are
read. The labels themselves are built locally from the same sort folders.

Edit the CONFIG block, then:  python ml_sector_label_features.py   (takes a minute or two on the login node)
"""

import glob
import os

import pandas as pd

# ---- CONFIG ----
PRED_DIR = '/fred/oz335/hroxburg/dev/ml_classifier/v5/sector_tests/predictions'   # ml_classify_sector.py's OUT_DIR
SORT_DIRS = ['/fred/oz335/hroxburg/dev/ml_classifier/v5/sector_tests/predictions/sort_sample/sort_sort_sample']
EVENTS_CSVS = []            # extra event lists, e.g. a targeted batch before it is sorted
FEATURE_VERSION = 5
OUT_FILE = '/fred/oz335/hroxburg/dev/ml_classifier/v5/sector_tests/S54_labelled_features_v5.csv.gz'
# ----------------

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']


def main():
    for d in SORT_DIRS:
        n = len(glob.glob(f'{d}/*/events.csv'))
        if n == 0:
            raise SystemExit(f'No */events.csv in {d} -- check the path (and its upper/lower case).')
        print(f'{d}: {n} label folders')
    keys = [pd.read_csv(f)[KEY_COLS] for d in SORT_DIRS for f in glob.glob(f'{d}/*/events.csv')]
    keys += [pd.read_csv(f)[KEY_COLS] for f in EVENTS_CSVS]
    keys = pd.concat(keys, ignore_index=True).drop_duplicates()
    print(f'{len(keys)} events in {keys.groupby(["sector", "camera", "ccd", "cut"]).ngroups} cuts')

    rows, missing = [], []
    for (s, cam, ccd, cut), k in keys.groupby(['sector', 'camera', 'ccd', 'cut']):
        path = f'{PRED_DIR}/S{s}/S{s}C{cam}C{ccd}C{cut}_features_v{FEATURE_VERSION}.csv.gz'
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            continue
        f = pd.read_csv(path, low_memory=False)
        rows.append(f.merge(k[['objid', 'eventid']], on=['objid', 'eventid']))
    if missing:
        print(f'{len(missing)} features files missing, e.g. {missing[:3]}')
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_FILE, index=False, compression='gzip')
    print(f'{len(out)} of {len(keys)} events -> {OUT_FILE}')


if __name__ == '__main__':
    main()

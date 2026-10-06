"""
For the tag-check events, pull the values the pipeline's asteroid rules used (com_motion, gaussian_score,
asteroid_id, ...) from their cuts' detected_events.csv, so we can see which rule tagged the non-asteroids.
Runs on the cluster in a minute or so; copy OUT_FILE down to development/ml_data/tag_check/.

Edit the CONFIG block, then:  python ml_tagcheck_rules.py
"""

import pandas as pd
from tessellate.tools import load_table

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
EVENTS_CSV = '/fred/oz335/hroxburg/dev/ml_classifier/tag_check/tag_check.csv'
OUT_FILE = '/fred/oz335/hroxburg/dev/ml_classifier/tag_check/tag_check_rules.csv'
# ----------------

COLS = ['objid', 'eventid', 'frame_bin', 'classification', 'com_motion', 'gaussian_score', 'asteroid_id',
        'frame_duration', 'frame_start', 'frame_end', 'psf_like', 'lc_sig_max', 'flux_sign', 'n_detections',
        'total_events']


def main():
    keys = pd.read_csv(EVENTS_CSV)
    rows = []
    for (sector, cam, ccd, cut), k in keys.groupby(['sector', 'camera', 'ccd', 'cut']):
        ev = load_table(f'{DATA_PATH}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of64/detected_events.csv')
        ev = ev[[c for c in COLS if c in ev]].merge(k[['objid', 'eventid', 'frame_bin']])
        rows.append(ev.assign(sector=sector, camera=cam, ccd=ccd, cut=cut))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_FILE, index=False)
    print(f'{len(out)} of {len(keys)} events -> {OUT_FILE}')


if __name__ == '__main__':
    main()

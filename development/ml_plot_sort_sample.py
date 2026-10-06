"""
Plot the ML sort sample on the cluster, ready for tools.manual_sort.

Reads sort_sample.csv (from ml_pick_sort_sample.py: event keys only), fetches each event's full row from its
cut's detected_events.csv, and saves
  EVENTS_OUT   the full rows -- the csv to give manual_sort
  IMAGE_DIR    one S{s}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png per event (nav.plot_lc, as in the sig10 sorts)
Images already made are skipped, so it can be stopped and rerun.
Events that fail to plot: full tracebacks go to IMAGE_DIR/plot_errors.txt, and the run ends with a count of
each distinct error (type, message and where it was raised).

Edit the CONFIG block, then:  python ml_plot_sort_sample.py
Then sort (with a GROUPS key for 'Unsure'), e.g.
  from tessellate.tools import manual_sort
  manual_sort(EVENTS_OUT, image_dir=IMAGE_DIR, sort_dir=SORT_DIR)
"""

import os
import traceback

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SAMPLE_CSV = '/fred/oz335/hroxburg/dev/ml_classifier/sort_sample/sort_sample.csv'
EVENTS_OUT = '/fred/oz335/hroxburg/dev/ml_classifier/sort_sample/sort_sample_events.csv'
IMAGE_DIR = '/fred/oz335/hroxburg/dev/ml_classifier/sort_sample/images'
EXTERNAL_PHOT = True
MPL_CACHE = '/fred/oz335/hroxburg/.matplotlib'   # matplotlib's config + LaTeX cache, instead of ~/.cache/matplotlib:
                                                 # a full home quota makes every LaTeX label fail
# ----------------

if MPL_CACHE:
    os.makedirs(MPL_CACHE, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = MPL_CACHE         # must be set before matplotlib is imported

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tessellate import Navigator  # noqa: E402

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']


def full_rows(sample):
    rows = []
    for (sector, cam, ccd, cut), keys in sample.groupby(['sector', 'camera', 'ccd', 'cut']):
        path = f'{DATA_PATH}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of64/detected_events.csv'
        ev = pd.read_csv(path)
        ev = ev.merge(keys[['objid', 'eventid', 'frame_bin']], on=['objid', 'eventid', 'frame_bin'])
        rows.append(ev)
    return pd.concat(rows, ignore_index=True)


def main():
    sample = pd.read_csv(SAMPLE_CSV)
    if os.path.exists(EVENTS_OUT):
        events = pd.read_csv(EVENTS_OUT)
    else:
        events = full_rows(sample)
        events.to_csv(EVENTS_OUT, index=False)
    missing = len(sample) - len(events)
    print(f'{len(events)} events' + (f' ({missing} not found in detected_events)' if missing else ''))

    os.makedirs(IMAGE_DIR, exist_ok=True)
    failures = []
    for (sector, cam, ccd), grp in events.groupby(['sector', 'camera', 'ccd']):
        nav = Navigator(sector, cam, ccd)
        for _, event in tqdm(grp.iterrows(), total=len(grp), desc=f'S{sector} C{cam} C{ccd}', ascii=True):
            name = f'S{event.sector}C{event.camera}C{event.ccd}C{event.cut}O{event.objid}E{event.eventid}.png'
            if os.path.exists(f'{IMAGE_DIR}/{name}'):
                continue
            try:
                nav.plot_lc(event, external_phot=EXTERNAL_PHOT, save_combined_path=IMAGE_DIR, verbose=False)
            except Exception as e:
                frames = traceback.extract_tb(e.__traceback__)
                where = ' <- '.join(f'{os.path.basename(f.filename)}:{f.lineno} {f.name}' for f in reversed(frames[-4:]))
                print(f'  {name}: not plotted ({type(e).__name__}: {e})\n      at {where}')
                failures.append({'event': name, 'error': f'{type(e).__name__}: {str(e)[:120]}',
                                 'raised_at': f'{frames[-1].filename}:{frames[-1].lineno}',
                                 'traceback': traceback.format_exc()})
    n_png = len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
    print(f'{n_png} images in {IMAGE_DIR}')
    if failures:
        fail = pd.DataFrame(failures)
        with open(f'{IMAGE_DIR}/plot_errors.txt', 'w') as f:
            for row in failures:
                f.write(f"===== {row['event']}\n{row['traceback']}\n")
        print(f'\n{len(fail)} events not plotted (full tracebacks in {IMAGE_DIR}/plot_errors.txt). Distinct errors:')
        print(fail.groupby(['error', 'raised_at']).size().sort_values(ascending=False).to_string())


if __name__ == '__main__':
    main()

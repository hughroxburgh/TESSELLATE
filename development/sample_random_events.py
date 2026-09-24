"""
Draw a uniformly random sample of the events nobody has looked at yet, and plot
them for manual sorting. Run this on the cluster (the plots need the flux cubes).

A sector's events fall into three groups:
  tagged     the pipeline's Junk / CosmicRay / Asteroid tags (trusted as labels)
  sorted     events already sorted by hand (found_flares.csv / non_flares.csv)
  rest       everything else -- the pool this script samples from
Groups 1 and 2 are covered completely, so a random sample of the rest is enough
to get unbiased numbers for the whole sector. The group sizes are written to
sample_summary.txt next to the sample.

The draw is a fixed-seed shuffle of the pool, so raising N_SAMPLE later keeps the
events already drawn and adds new ones (only the new ones get plotted).

Edit the CONFIG block, then:  python sample_random_events.py
Then sort the images with tools.manual_sort into their own sort folder (e.g.
sort_random) -- keep it separate from other sorts, its value is being random.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTOR = 55
CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]
N_CUTS = 64

N_SAMPLE = 300
SEED = 55
FRAME_BIN = 1               # the manual sort shows frame_bin 1 (coarser bins inherit its labels); None = every bin
TAGGED = ['Junk', 'CosmicRay', 'Asteroid']   # pipeline classifications trusted as labels, so never sampled
SORTED_CSVS = ['/fred/oz335/hroxburg/dev/final_localisation/found_flares.csv',   # events already sorted by hand
               '/fred/oz335/hroxburg/dev/final_localisation/non_flares.csv']

OUT_DIR = f'/fred/oz335/hroxburg/dev/random_sample/Sector{SECTOR}'
PLOT = True                 # draw a PNG per sampled event (same plot as plot_events.py)
# ----------------

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']


def load_sector_events():
    tables = []
    for cam in CAMS:
        for ccd in CCDS:
            for cut in range(1, N_CUTS + 1):
                path = f'{DATA_PATH}/Sector{SECTOR}/Cam{cam}/Ccd{ccd}/Cut{cut}of{N_CUTS}/detected_events.csv'
                if os.path.exists(path):
                    tables.append(pd.read_csv(path, low_memory=False))   # known_asteroid_designation mixes numbers and names
    if not tables:
        raise FileNotFoundError(f'No detected_events.csv found for Sector {SECTOR} under {DATA_PATH}')
    return pd.concat(tables, ignore_index=True)


def draw_sample(events):
    """Returns (sample, summary lines)."""
    if FRAME_BIN is not None:
        events = events[events.frame_bin == FRAME_BIN]
    tagged = events.classification.isin(TAGGED)

    sorted_keys = pd.concat([pd.read_csv(f, usecols=KEY_COLS)[KEY_COLS] for f in SORTED_CSVS], ignore_index=True)
    sorted_keys = set(map(tuple, sorted_keys.drop_duplicates().to_numpy()))
    is_sorted = np.array([tuple(k) in sorted_keys for k in events[KEY_COLS].to_numpy()])

    pool = events[~tagged & ~is_sorted].sort_values(KEY_COLS)
    order = np.random.default_rng(SEED).permutation(len(pool))
    sample = pool.iloc[order[:N_SAMPLE]].sort_values(KEY_COLS)   # cut order: plot_lc reloads the cube per cut

    lines = [f'Sector {SECTOR}, frame_bin {FRAME_BIN if FRAME_BIN is not None else "all"}, seed {SEED}',
             f'  all events            {len(events)}',
             f'  tagged by pipeline    {tagged.sum()}  ({", ".join(TAGGED)})',
             f'  sorted by hand        {(is_sorted & ~tagged).sum()}  (of {len(sorted_keys)} in SORTED_CSVS)',
             f'  rest (the pool)       {len(pool)}',
             f'  sampled               {len(sample)}  (fraction {len(sample) / max(len(pool), 1):.4f} of the pool)']
    if (is_sorted & tagged).any():
        lines.append(f'  note: {(is_sorted & tagged).sum()} sorted events are also tagged; counted as tagged')
    return sample, lines


def plot_sample(sample, image_dir):
    from tessellate import Navigator

    os.makedirs(image_dir, exist_ok=True)
    failed = 0
    for (cam, ccd), grp in sample.groupby(['camera', 'ccd']):
        nav = Navigator(SECTOR, cam, ccd, data_path=DATA_PATH)
        for _, event in tqdm(grp.iterrows(), total=len(grp), desc=f'Camera {cam}, CCD {ccd}', ascii=True):
            name = f'S{event.sector}C{event.camera}C{event.ccd}C{event.cut}O{event.objid}E{event.eventid}.png'
            if os.path.exists(f'{image_dir}/{name}'):
                continue
            try:
                nav.plot_lc(event, external_phot=True, save_combined_path=image_dir, verbose=False)
            except Exception as e:
                failed += 1
                print(f'  {name}: not plotted ({type(e).__name__}: {e})')
    if failed:
        print(f'{failed} events could not be plotted.')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sample, lines = draw_sample(load_sector_events())
    sample.to_csv(f'{OUT_DIR}/random_sample.csv', index=False)
    with open(f'{OUT_DIR}/sample_summary.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {OUT_DIR}/random_sample.csv')

    if PLOT:
        plot_sample(sample, f'{OUT_DIR}/images')
        print(f'Images in {OUT_DIR}/images')


if __name__ == '__main__':
    main()

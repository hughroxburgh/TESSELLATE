"""
Gaia neighbours of every confident flare, for calibrating the localisation error model and its outlier rate.
Run on the cluster (needs each cut's detected_events and local_gaia_cat.csv).

Edit the CONFIG block, then:  python localisation_neighbours.py

SLURM = True submits ONE job that works through every cut with N_JOBS workers (log in OUT_DIR/slurm_logs);
SLURM = False runs it right here.

Events: filter_events(classification='Flare', min_probability=MIN_P_FLARE, frame_bin=1, psf_like=MIN_PSF_LIKE,
starkiller=None) -- every confident PSF-like flare, matched to a star or not.

For each event, the K nearest Gaia stars within RADIUS_PX of its position (pixel offsets dx, dy = star - event, in
the cut's WCS pixel frame, the same frame as the detector's crossmatch, and magnitudes), plus star counts within
1 / 2 / 3 / 5 px. The same is recorded at N_SHIFTS positions moved SHIFT_PX_MIN - SHIFT_PX_MAX px in random
directions (kept inside the cut): how often a random spot lands near a star, i.e. chance alignment.

Output, OUT_DIR/S{sector}_neighbours.parquet (one row per event) and OUT_DIR/S{sector}_shifted.parquet (one row per
shifted position, with the event's keys and shift number):
  event columns (EVENT_COLS), x_wcs / y_wcs (event position in the WCS pixel frame),
  n1px_*, n2px_*, n3px_*, n5px_*: stars within that radius (all, and G < 17)
  nb{k}_dx, nb{k}_dy, nb{k}_r, nb{k}_G, nb{k}_RP, nb{k}_source for k = 1..K (nearest first; NaN if fewer)
Isolation and the fits are done afterwards, locally, from these tables.
"""

import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTORS = [53, 54, 55]
CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]
CUTS = range(1, 65)
N_CUTS = 64

MIN_P_FLARE = 0.9
MIN_PSF_LIKE = 0.5           # PSF-like events only: the others have no centroid_err
RADIUS_PX = 6.0              # neighbours searched within this many pixels
K = 8                        # nearest neighbours kept per position
N_SHIFTS = 3                 # shifted (random) positions per event
SHIFT_PX_MIN, SHIFT_PX_MAX = 6.0, 12.0
CUT_SIZE_PX = 264            # positions moved outside [0, CUT_SIZE_PX) are redrawn
SEED = 0

EVENT_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid', 'xcentroid', 'ycentroid', 'xccd', 'yccd', 'ra',
              'dec', 'centroid_err', 'snr_psf', 'psf_like', 'psf_det_sep', 'psf_pinned', 'psf_stacked',
              'lc_sig_max', 'image_sig_max', 'p_Flare', 'frame_duration', 'mjd_max', 'gal_b', 'flux_max', 'mag_min',
              'gaia_id', 'nearest_gaia_id', 'nearest_gaia_dx', 'nearest_gaia_dy', 'var_type', 'source_mask']

SLURM = True                 # True = submit one job; False = run right here
N_JOBS = 16                  # cuts done in parallel (and, with SLURM, the CPUs requested)
JOB_TIME = '04:00:00'
MEM_PER_CPU_GB = 4
ACCOUNT = 'oz335'
OUT_DIR = '/fred/oz335/hroxburg/dev/localisation_neighbours'
# ----------------


def _neighbours(tree, gx, gy, gG, gRP, gsrc, x, y):
    """Columns describing the Gaia stars around positions (x, y)."""
    out = {}
    pts = np.c_[x, y]
    for r in (1, 2, 3, 5):
        idx = tree.query_ball_point(pts, r)
        out[f'n{r}px_all'] = [len(i) for i in idx]
        out[f'n{r}px_G17'] = [int(np.sum(gG[i] < 17)) for i in idx]
    d, i = tree.query(pts, k=K, distance_upper_bound=RADIUS_PX)
    d, i = np.atleast_2d(d), np.atleast_2d(i)
    ok = np.isfinite(d)
    j = np.where(ok, i, 0)
    for k in range(K):
        m = ok[:, k]
        out[f'nb{k + 1}_dx'] = np.where(m, gx[j[:, k]] - x, np.nan)
        out[f'nb{k + 1}_dy'] = np.where(m, gy[j[:, k]] - y, np.nan)
        out[f'nb{k + 1}_r'] = np.where(m, d[:, k], np.nan)
        out[f'nb{k + 1}_G'] = np.where(m, gG[j[:, k]], np.nan)
        out[f'nb{k + 1}_RP'] = np.where(m, gRP[j[:, k]], np.nan)
        out[f'nb{k + 1}_source'] = np.where(m, gsrc[j[:, k]].astype(str), '')
    return pd.DataFrame(out)


def cut_job(sector, cam, ccd, cut):
    """(events table, shifted table) for one cut, or None if there's nothing to do."""
    from scipy.spatial import cKDTree
    from tessellate import Navigator
    from tessellate.tools import table_exists

    path = f'{DATA_PATH}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{N_CUTS}'
    if not (table_exists(f'{path}/detected_events.csv') and os.path.exists(f'{path}/local_gaia_cat.csv')):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nav = Navigator(sector, cam, ccd, data_path=DATA_PATH)
        nav.gather_results(cut=cut, sources=False, objects=False)
        if nav.events is None or 'p_Flare' not in nav.events:
            return None
        ev = nav.filter_events(cut=cut, classification='Flare', min_probability=MIN_P_FLARE, frame_bin=1,
                               psf_like=MIN_PSF_LIKE, starkiller=None)
        ev = ev[np.isfinite(ev.centroid_err)]
        if len(ev) == 0:
            return None

        gaia = pd.read_csv(f'{path}/local_gaia_cat.csv')
        gx, gy = nav.wcs.all_world2pix(gaia.ra.values, gaia.dec.values, 0)
        ex, ey = nav.wcs.all_world2pix(ev.ra.values, ev.dec.values, 0)   # as the detector's crossmatch does
    gx, gy, ex, ey = (np.asarray(a, float) for a in (gx, gy, ex, ey))
    gG = gaia.Gmag.to_numpy(float)
    gRP = gaia.RPmag.to_numpy(float) if 'RPmag' in gaia else np.full(len(gaia), np.nan)
    gsrc = gaia.Source.to_numpy()
    tree = cKDTree(np.c_[gx, gy])

    keys = ev[[c for c in EVENT_COLS if c in ev]].reset_index(drop=True)
    keys['sector'], keys['camera'], keys['ccd'], keys['cut'] = sector, cam, ccd, cut
    keys['x_wcs'], keys['y_wcs'] = ex, ey
    events = pd.concat([keys, _neighbours(tree, gx, gy, gG, gRP, gsrc, ex, ey)], axis=1)

    # -- random nearby positions: chance alignment -- #
    rng = np.random.default_rng([SEED, sector, cam, ccd, cut])
    sx, sy, sid = [], [], []
    for n in range(N_SHIFTS):
        x, y = ex.copy(), ey.copy()
        todo = np.ones(len(x), bool)
        for _ in range(100):                       # redraw positions that fall off the cut
            m = todo.sum()
            if m == 0:
                break
            ang = rng.uniform(0, 2 * np.pi, m)
            dist = rng.uniform(SHIFT_PX_MIN, SHIFT_PX_MAX, m)
            x[todo] = ex[todo] + dist * np.cos(ang)
            y[todo] = ey[todo] + dist * np.sin(ang)
            todo = (x < 0) | (x >= CUT_SIZE_PX) | (y < 0) | (y >= CUT_SIZE_PX)
        sx.append(x); sy.append(y); sid.append(np.full(len(x), n))
    sx, sy, sid = np.concatenate(sx), np.concatenate(sy), np.concatenate(sid)
    base = pd.concat([keys[['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid', 'centroid_err', 'snr_psf']]] * N_SHIFTS,
                     ignore_index=True)
    base['shift'], base['x_shift'], base['y_shift'] = sid, sx, sy
    shifted = pd.concat([base, _neighbours(tree, gx, gy, gG, gRP, gsrc, sx, sy)], axis=1)
    return events, shifted


def run():
    from joblib import Parallel, delayed
    from tqdm import tqdm
    os.makedirs(OUT_DIR, exist_ok=True)
    for sector in SECTORS:
        t = time.time()
        jobs = [(sector, cam, ccd, cut) for cam in CAMS for ccd in CCDS for cut in CUTS]
        res = Parallel(n_jobs=N_JOBS)(delayed(cut_job)(*j) for j in tqdm(jobs, desc=f'Sector {sector}'))
        res = [r for r in res if r is not None]
        if not res:
            print(f'Sector {sector}: nothing found')
            continue
        events = pd.concat([r[0] for r in res], ignore_index=True)
        shifted = pd.concat([r[1] for r in res], ignore_index=True)
        events.to_parquet(f'{OUT_DIR}/S{sector}_neighbours.parquet', index=False)
        shifted.to_parquet(f'{OUT_DIR}/S{sector}_shifted.parquet', index=False)
        print(f'Sector {sector}: {len(events)} flares from {len(res)} cuts, {len(shifted)} shifted positions '
              f'({time.time() - t:.0f}s) -> {OUT_DIR}', flush=True)


def submit():
    log_dir = f'{OUT_DIR}/slurm_logs'
    os.makedirs(log_dir, exist_ok=True)
    script = f'{OUT_DIR}/localisation_neighbours.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                '#SBATCH --job-name=loc_neighbours\n'
                f'#SBATCH --output={log_dir}/%j_out.txt\n'
                f'#SBATCH --error={log_dir}/%j_err.txt\n'
                '#SBATCH --ntasks=1\n'
                f'#SBATCH --cpus-per-task={N_JOBS}\n'
                f'#SBATCH --mem-per-cpu={MEM_PER_CPU_GB}G\n'
                f'#SBATCH --time={JOB_TIME}\n'
                f'#SBATCH --account={ACCOUNT}\n\n'
                'export PYTHONUNBUFFERED=1\n'
                'export OMP_NUM_THREADS=1\n'
                f'{sys.executable} {os.path.abspath(__file__)}\n')
    result = subprocess.run(['sbatch', script], capture_output=True, text=True)
    if result.returncode:
        sys.exit(f'sbatch failed: {result.stderr.strip()}')
    print(f'{result.stdout.strip()} -- logs in {log_dir}')


if __name__ == '__main__':
    if SLURM and 'SLURM_JOB_ID' not in os.environ:
        submit()
    else:
        run()

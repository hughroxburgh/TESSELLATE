"""
Extract tessellate.ml_classifier features for a sector. Run this on the
cluster, where the reduced flux cubes live (~1 GB per cut); the output table
is small enough to copy anywhere for training.

Edit the CONFIG block, then:  python ml_extract_features.py

With SLURM = True, running it on the login node submits it as a SLURM job
(logs in OUT_DIR/slurm_logs); inside the job it does the work. With SLURM =
False it runs right where you start it.

With CACHE_DIR set, every finished cut is saved there, so if the job dies or
runs out of time, running the script again carries on where it stopped.
"""

import os
import subprocess
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
N_JOBS = 16                 # parallel workers (and, with SLURM, the CPUs requested)
CROSSMATCH = True           # Gaia / variable-catalogue features (needs the WCS + local catalogues)
FRAME_STATS = True          # per-frame noise pass over each cube (~10 s per cut)
OVERWRITE = False           # True = recompute cuts already in CACHE_DIR

SLURM = True                # True = submit as a SLURM job; False = run here
TIME = '02:00:00'           # wall time (a sector should take well under an hour at 16 CPUs)
MEM_PER_CPU_GB = 4          # cubes are memory-mapped, so each worker needs little
ACCOUNT = 'oz335'
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


def submit():
    """Write a batch script that runs this file with the same Python, and sbatch it."""
    log_dir = f'{OUT_DIR}/slurm_logs'
    os.makedirs(log_dir, exist_ok=True)
    script = f'{OUT_DIR}/S{SECTOR}_extract_features.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                f'#SBATCH --job-name=ml_features_S{SECTOR}\n'
                f'#SBATCH --output={log_dir}/%j_out.txt\n'
                f'#SBATCH --error={log_dir}/%j_err.txt\n'
                '#SBATCH --ntasks=1\n'
                f'#SBATCH --cpus-per-task={N_JOBS}\n'
                f'#SBATCH --mem-per-cpu={MEM_PER_CPU_GB}G\n'
                f'#SBATCH --time={TIME}\n'
                f'#SBATCH --account={ACCOUNT}\n\n'
                'export PYTHONUNBUFFERED=1\n'
                f'{sys.executable} {os.path.abspath(__file__)}\n')
    result = subprocess.run(['sbatch', script], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'sbatch failed: {result.stderr.strip()}')
    print(result.stdout.strip())
    print(f'Logs: {log_dir}   (check progress with: squeue -u $USER)')


if __name__ == '__main__':
    if SLURM and 'SLURM_JOB_ID' not in os.environ:   # on the login node: submit; inside the job: run
        submit()
    else:
        main()

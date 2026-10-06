"""
Extract tessellate.ml_classifier features for one or more sectors. Run this on
the cluster, where the reduced flux cubes live (~1 GB per cut). Every cut's
features are written to its sector's cache folder as one csv; a whole sector is
tens of millions of events, far too many for one table, so
ml_collect_features.py then picks the rows needed for training into a file
small enough to copy down.

Two modes:
  LABELS_CSV set (manual_labels.csv from ml_build_labels.py): only the cuts
    holding a labelled event, and in those cuts only
      - the labelled events,
      - up to TAGGED_PER_CUT of each pipeline tag (Junk / CosmicRay / Asteroid): weak labels,
      - up to VARIABLES_PER_CUT events the pipeline matched to a variable-star catalogue
        (classification V...): candidates for Variable labels,
      - up to RANDOM_PER_CUT of everything else: background.
    The picks are a fixed random choice per cut, saved with the reason for each
    in <cache>/S{sector}_extract_events.csv. Much faster than whole sectors.
  LABELS_CSV = None: every event of every cut (or those in EVENTS_CSV), with
    at most TAGGED_PER_CUT of each pipeline tag.

Edit the CONFIG block, then:  python ml_extract_features.py

With SLURM = True, running it on the login node submits it as a SLURM job
(logs in OUT_DIR/slurm_logs); inside the job it does the work. With SLURM =
False it runs right where you start it.

If the job dies or runs out of time, running the script again carries on
where it stopped: finished cuts are skipped, and a file cut short mid-write is
noticed (fewer rows than the cut should have) and redone.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
import tessellate.ml_classifier
from tessellate.tools import load_table, table_exists
from tessellate.ml_classifier import FEATURE_VERSION, PIPELINE_CLASSES, _cut_path, build_feature_table

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTORS = list(range(27, 40)) + [55]   # done one after another in the same job
CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]
CUTS = range(1, 65)                     # all 64 cuts (with LABELS_CSV, only the labelled ones among these)

OUT_DIR = '/fred/oz335/hroxburg/dev/ml_classifier'
CACHE_NAME = 'S{sector}_feature_cache_labelled'   # per-sector cache folder in OUT_DIR. Keep it different from a
                                                  # whole-sector cache: a cut's file only holds the events asked for

LABELS_CSV = '/fred/oz335/hroxburg/dev/ml_classifier/manual_labels_all.csv'   # None = every event
VARIABLES_PER_CUT = 50      # with LABELS_CSV: catalogue-variable events per cut (None = all)
RANDOM_PER_CUT = 50         # with LABELS_CSV: other untagged events per cut
SEED = 0

EVENTS_CSV = None           # without LABELS_CSV: a csv of events (sector/camera/ccd/cut/objid/eventid) to restrict
                            # to. None = every event.
N_JOBS = 16                 # parallel workers (and, with SLURM, the CPUs requested)
CROSSMATCH = False          # Gaia / variable-catalogue features (needs the WCS + local catalogues). Off: stage 1
                            # uses no localisation, and S55's radial-branch columns break it
FRAME_STATS = True          # per-frame noise pass over each cube (~8 s per cut)
TAGGED_PER_CUT = 100        # per cut, at most this many events of each pipeline tag (Junk/CosmicRay/Asteroid) get
                            # features -- they're only weak labels, and ~45% of a cut. None = all of them
OVERWRITE = False           # True = recompute cuts already in the cache (do this if you change the per-cut numbers)

SLURM = True                # True = submit as a SLURM job; False = run here
TIME = '08:00:00'           # wall time: labelled cuts take ~10-15 s each on 16 CPUs; whole sectors ~6.5 h each
                            # (S55: ~23 s per cut). If it runs out, run the script again: it carries on
MEM_PER_CPU_GB = 4          # cubes are memory-mapped, so each worker needs little
ACCOUNT = 'oz335'
# ----------------


def pick_cut_events(sector, cam, ccd, cut, labelled):
    """The events of one labelled cut to extract, with the reason each was picked."""
    path = f'{_cut_path(DATA_PATH, sector, cam, ccd, cut)}/detected_events.csv'
    if not table_exists(path):
        return None
    ev = load_table(path, columns=['objid', 'eventid', 'classification'])
    rng = np.random.default_rng([SEED, sector, cam, ccd, cut])
    cls = ev['classification'].astype(str)

    is_lab = ev.set_index(['objid', 'eventid']).index.isin(labelled.set_index(['objid', 'eventid']).index)
    is_tag = cls.isin(PIPELINE_CLASSES).to_numpy() & ~is_lab
    is_var = cls.str.startswith('V').to_numpy() & ~is_lab
    is_other = ~is_lab & ~is_tag & ~is_var

    def sample(mask, k):
        idx = np.flatnonzero(mask)
        return idx if k is None or len(idx) <= k else np.sort(rng.choice(idx, k, replace=False))

    picked = [(np.flatnonzero(is_lab), 'labelled'), (sample(is_var, VARIABLES_PER_CUT), 'catalogue_variable'),
              (sample(is_other, RANDOM_PER_CUT), 'random')]
    for c in PIPELINE_CLASSES:          # cap each tag separately
        picked.append((sample(is_tag & (cls == c).to_numpy(), TAGGED_PER_CUT), 'tagged'))
    rows = [ev.iloc[idx][['objid', 'eventid']].assign(picked_as=why) for idx, why in picked if len(idx)]
    return pd.concat(rows).assign(sector=sector, camera=cam, ccd=ccd, cut=cut)


def labelled_events(sector, labels):
    from joblib import Parallel, delayed
    lab = labels[(labels.sector == sector) & labels.camera.isin(CAMS) & labels.ccd.isin(CCDS) & labels.cut.isin(CUTS)]
    groups = list(lab.groupby(['camera', 'ccd', 'cut']))
    parts = Parallel(n_jobs=N_JOBS)(delayed(pick_cut_events)(sector, cam, ccd, cut, g) for (cam, ccd, cut), g in groups)
    missing = [key for key, p in zip([k for k, _ in groups], parts) if p is None]
    if missing:
        print(f'  {len(missing)} labelled cuts have no detected_events.csv, e.g. cam/ccd/cut {missing[:3]}')
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = pd.read_csv(LABELS_CSV) if LABELS_CSV else None
    events_all = pd.read_csv(EVENTS_CSV) if EVENTS_CSV and not LABELS_CSV else None
    print(f'Using {tessellate.ml_classifier.__file__} (feature version {FEATURE_VERSION})', flush=True)

    for sector in SECTORS:
        cache_dir = f'{OUT_DIR}/{CACHE_NAME.format(sector=sector)}'
        os.makedirs(cache_dir, exist_ok=True)
        start = time.time()
        if labels is not None:
            events = labelled_events(sector, labels)
            if events is None:
                print(f'Sector {sector}: no labelled cuts found; skipped', flush=True)
                continue
            events.to_csv(f'{cache_dir}/S{sector}_extract_events.csv', index=False)
            counts = events.picked_as.value_counts().to_dict()
            print(f'Sector {sector}: {events.groupby(["camera", "ccd", "cut"]).ngroups} cuts, {counts}', flush=True)
            max_tagged = None           # already capped above (labelled events are never dropped)
        else:
            events, max_tagged = events_all, TAGGED_PER_CUT

        try:
            summary = build_feature_table(DATA_PATH, sector, cams=CAMS, ccds=CCDS, cuts=CUTS, events=events,
                                          cache_dir=cache_dir, overwrite=OVERWRITE, n_jobs=N_JOBS,
                                          return_table=False,
                                          config={'crossmatch': CROSSMATCH, 'frame_stats': FRAME_STATS,
                                                  'max_tagged': max_tagged})
        except ValueError as e:
            print(f'Sector {sector}: {e}', flush=True)
            continue
        summary.to_csv(f'{cache_dir}/S{sector}_cut_summary.csv', index=False)
        print(f'Sector {sector}: {int(summary.n_events.sum())} events in {len(summary)} cuts, '
              f'{(time.time() - start) / 60:.1f} min -> {cache_dir}', flush=True)
    print('Next: ml_collect_features.py builds the training table from these files.')


def submit():
    """Write a batch script that runs this file with the same Python, and sbatch it."""
    log_dir = f'{OUT_DIR}/slurm_logs'
    os.makedirs(log_dir, exist_ok=True)
    name = f'S{SECTORS[0]}-{SECTORS[-1]}' if len(SECTORS) > 1 else f'S{SECTORS[0]}'
    script = f'{OUT_DIR}/{name}_extract_features.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                f'#SBATCH --job-name=ml_features_{name}\n'
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

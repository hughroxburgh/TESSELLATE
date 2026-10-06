"""
Classify every event of whole sectors with a trained stage-1 model
(tessellate.ml_classifier.EventClassifier). Run this on the cluster, where the
reduced flux cubes live.

Edit the CONFIG block, then:  python ml_classify_sector.py

It looks at what's already done:
  - cuts still to do  -> classifies them, one of three ways (CONFIG):
      ARRAY = True:  submits a SLURM job array, CUTS_PER_TASK cuts per 1-CPU task.
                     Good for a first pass over whole sectors (feature extraction,
                     minutes per cut) when the queue is quiet.
      ARRAY = False, SLURM = True:  submits ONE job that works through every
                     cut with N_JOBS workers. Better when the array tasks spend
                     longer queuing than working -- e.g. predicting from saved
                     features, seconds per cut.
      ARRAY = False, SLURM = False:  the same, right here (fine on the login
                     node for predict-only reruns with a few workers).
    Logs in OUT_DIR/slurm_logs.
  - nothing left to do -> says so; with COLLECT = True it also joins the per-cut
    files into one table per sector and prints a summary (predicted class against
    the pipeline's tags). Next step: ml_sector_sample.py.
If some cuts died or ran out of time, running it again redoes only those.

Per cut, in OUT_DIR/S{sector}/:
  S{s}C{cam}C{ccd}C{cut}_ml_predictions.csv  event keys, the pipeline's
      classification, p_Junk / p_CosmicRay / p_Asteroid / p_Flare / p_Variable,
      p_astrophysical, pred_class, margin, entropy, domain_out_frac and review
      (True = the model is unsure: its top probability is below REVIEW_MAX_P)
  S{s}C{cam}C{ccd}C{cut}_features_v{N}.csv.gz  the features (SAVE_FEATURES), so
      sorted events can later be used for training without re-extracting
A file only appears once it's complete, so a killed task leaves no half files. A cut whose features
file already exists reuses it, so a rerun only redoes the (fast) prediction.
"""

import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
import tessellate.ml_classifier
from tessellate.ml_classifier import FEATURE_VERSION, EventClassifier, _cut_path, extract_cut_features, load_cut_events

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTORS = [53, 54]
CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]
CUTS = range(1, 65)
FRAME_BINS = [1]            # the model is trained on frame bin 1 only

MODEL = '/fred/oz335/hroxburg/dev/ml_classifier/event_classifier_stage1_v5_D.joblib'
OUT_DIR = '/fred/oz335/hroxburg/dev/ml_classifier/predictions'
SAVE_FEATURES = True        # also keep each cut's features (~a few MB per cut, gzipped)
REVIEW_MAX_P = 0.75         # review = True when the most likely class has p below this
CROSSMATCH = False          # the model doesn't use the Gaia / variable-catalogue features
COLLECT = False             # True = when every cut is done, also join them into one S{s}_ml_predictions.csv.gz
                            # (~1 GB per sector, slow to write). ml_sector_sample.py reads the per-cut files and
                            # writes a slimmer table with lc_sig_max, so this is rarely needed
COMPATIBLE = {(5, 6)}       # (model, installed) feature versions that are safe together: v6 only ADDS ctx_lsl_*,
                            # every v5 feature is unchanged

ARRAY = False               # True = job array (see the docstring); False = one job / here with N_JOBS workers
SLURM = True                # with ARRAY = False: True = submit one job, False = run right here
N_JOBS = 16                 # with ARRAY = False: cuts done in parallel (and, with SLURM, the CPUs requested)
JOB_TIME = '08:00:00'       # with ARRAY = False and SLURM: wall time of the one job. Predict-only: ~minutes;
                            # extracting: ~2-10 min per cut / N_JOBS (a full sector ~3-5 h on 16 CPUs)
MEM_PER_CPU_GB = 4          # with ARRAY = False and SLURM

CUTS_PER_TASK = 4           # with ARRAY: cuts done one after another by each task (keeps the array under MAX_ARRAY)
MAX_ARRAY = 1000            # SLURM's MaxArraySize is often 1001
MAX_RUNNING = 200           # with ARRAY: tasks running at once
TIME = '03:00:00'           # with ARRAY: wall time per task: ~30 ms per event locally (4.4k events in 133 s), so a
                            # crowded cut of 20k frame-bin-1 events is ~10-20 min
MEM_GB = 6                  # with ARRAY: per task; the cube is memory-mapped
ACCOUNT = 'oz335'
# ----------------


def pred_path(sector, cam, ccd, cut):
    return f'{OUT_DIR}/S{sector}/S{sector}C{cam}C{ccd}C{cut}_ml_predictions.csv'


def feat_path(sector, cam, ccd, cut):
    return f'{OUT_DIR}/S{sector}/S{sector}C{cam}C{ccd}C{cut}_features_v{FEATURE_VERSION}.csv.gz'


def classify_cut(sector, cam, ccd, cut, model):
    """Features + predictions for the frame-bin events of one cut; returns the number of events (None = no cut)."""
    if not os.path.exists(f'{_cut_path(DATA_PATH, sector, cam, ccd, cut)}/detected_events.csv'):
        return None
    events = load_cut_events(DATA_PATH, sector, cam, ccd, cut)
    events = events[events.frame_bin.isin(FRAME_BINS)]
    out = pred_path(sector, cam, ccd, cut)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not len(events):
        pd.DataFrame(columns=['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']).to_csv(out, index=False)
        return 0
    fpath = feat_path(sector, cam, ccd, cut)
    if os.path.exists(fpath):                   # saved by an earlier run (only complete files exist)
        feats = pd.read_csv(fpath, low_memory=False)
    else:
        feats = extract_cut_features(DATA_PATH, sector, cam, ccd, cut, events=events,
                                     config={'crossmatch': CROSSMATCH, 'frame_stats': True, 'max_tagged': None})
    missing = [c for c in model.feature_names_ if c not in feats]
    if missing:
        raise ValueError(f'{len(missing)} model features not extracted, e.g. {missing[:3]}')
    if SAVE_FEATURES and not os.path.exists(fpath):
        feats.to_csv(f'{fpath}.part', index=False, compression='gzip')
        os.replace(f'{fpath}.part', fpath)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pred = model.predict(feats)
    p_cols = [c for c in pred if c.startswith('p_') and c != 'p_astrophysical']
    pred['review'] = pred[p_cols].max(axis=1) < REVIEW_MAX_P
    pred.to_csv(f'{out}.part', index=False)
    os.replace(f'{out}.part', out)              # the file only appears once it's complete
    return len(pred)


def all_cuts():
    return [(s, cam, ccd, cut) for s in SECTORS for cam in CAMS for ccd in CCDS for cut in CUTS]


def load_model(verbose=True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # EventClassifier.load warns on any version difference
        model = EventClassifier.load(MODEL)
    # A model saved with scikit-learn 1.3 lacks the _preprocessor attribute that 1.4+ checks when predicting
    # (it's only set for categorical features, which this model doesn't have): None = no preprocessing, as in 1.3
    if not hasattr(model.model_, '_preprocessor'):
        model.model_._preprocessor = None
    version = getattr(model, 'version', None)
    if (version, FEATURE_VERSION) in COMPATIBLE:
        if verbose:
                print(f'Model feature version {version} on installed version {FEATURE_VERSION}: compatible', flush=True)
    elif version != FEATURE_VERSION:
        sys.exit(f'The model was trained on feature version {getattr(model, "version", None)}, but the installed '
                 f'tessellate ({tessellate.ml_classifier.__file__}) computes version {FEATURE_VERSION}. '
                 'Install the matching tessellate, or use a model trained on this version.')
    return model


def process_cut(sector, cam, ccd, cut, model):
    """classify_cut with a log line; a failure is logged and skipped, not raised."""
    start = time.time()
    try:
        n = classify_cut(sector, cam, ccd, cut, model)
    except Exception as e:
        print(f'S{sector} C{cam} C{ccd} cut {cut}: FAILED ({type(e).__name__}: {e})', flush=True)
        return
    what = 'no detected_events.csv' if n is None else f'{n} events'
    print(f'S{sector} C{cam} C{ccd} cut {cut}: {what}, {time.time() - start:.0f} s', flush=True)


_MODEL = None


def _worker(sector, cam, ccd, cut):
    """One cut in a parallel worker; each worker process loads the model once."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(verbose=False)
    process_cut(sector, cam, ccd, cut, _MODEL)


def todo_cuts():
    return [c for c in all_cuts() if not os.path.exists(pred_path(*c))
            and os.path.exists(f'{_cut_path(DATA_PATH, *c)}/detected_events.csv')]


def run_here(todo):
    """Classify the cuts with N_JOBS parallel workers in this process, then collect if nothing is left."""
    from joblib import Parallel, delayed
    print(f'Using {tessellate.ml_classifier.__file__} (feature version {FEATURE_VERSION}), model {MODEL}', flush=True)
    print(f'{len(todo)} cuts to do with {N_JOBS} workers', flush=True)
    start = time.time()
    Parallel(n_jobs=N_JOBS)(delayed(_worker)(*c) for c in todo)
    left = todo_cuts()
    print(f'\nDone in {(time.time() - start) / 60:.1f} min; {len(left)} cuts still without predictions', flush=True)
    if left:
        print('Run the script again to retry them (their FAILED lines above say why).')
    elif COLLECT:
        collect()
    else:
        print('All cuts done. Next: ml_sector_sample.py')


def run_task():
    """Inside an array task: classify this task's cuts."""
    tasks = pd.read_csv(os.environ['ML_CLASSIFY_TASKS'])
    mine = tasks[tasks.task == int(os.environ['SLURM_ARRAY_TASK_ID'])]
    model = load_model()
    print(f'Using {tessellate.ml_classifier.__file__} (feature version {FEATURE_VERSION}), model {MODEL}', flush=True)
    for row in mine.itertuples():
        process_cut(row.sector, row.camera, row.ccd, row.cut, model)


def submit_job(todo):
    """Write a batch script that runs this file as ONE job with N_JOBS CPUs, and sbatch it."""
    log_dir = f'{OUT_DIR}/slurm_logs'
    os.makedirs(log_dir, exist_ok=True)
    name = '-'.join(str(s) for s in sorted({c[0] for c in todo}))
    script = f'{OUT_DIR}/classify_S{name}_{time.strftime("%Y%m%d_%H%M%S")}.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                f'#SBATCH --job-name=ml_classify_S{name}\n'
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
    if result.returncode != 0:
        sys.exit(f'sbatch failed: {result.stderr.strip()}')
    print(f'{len(todo)} cuts in one job with {N_JOBS} CPUs: {result.stdout.strip()}')
    print(f'Log: {log_dir}/<jobid>_out.txt')


def submit_array(todo):
    """Write the task list and a batch script for the cuts still to do, and sbatch it."""
    log_dir = f'{OUT_DIR}/slurm_logs'
    os.makedirs(log_dir, exist_ok=True)
    tasks = pd.DataFrame(todo, columns=['sector', 'camera', 'ccd', 'cut'])
    tasks['task'] = np.arange(len(tasks)) // CUTS_PER_TASK
    n_tasks = int(tasks.task.max()) + 1
    if n_tasks > MAX_ARRAY:
        sys.exit(f'{len(tasks)} cuts at {CUTS_PER_TASK} per task is {n_tasks} array tasks, more than MAX_ARRAY = '
                 f'{MAX_ARRAY}. Raise CUTS_PER_TASK (and TIME), or do fewer SECTORS at once.')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    task_file = f'{OUT_DIR}/classify_tasks_{stamp}.csv'
    tasks.to_csv(task_file, index=False)
    name = '-'.join(str(s) for s in sorted(tasks.sector.unique()))
    script = f'{OUT_DIR}/classify_S{name}_{stamp}.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                f'#SBATCH --job-name=ml_classify_S{name}\n'
                f'#SBATCH --output={log_dir}/%A_%a_out.txt\n'
                f'#SBATCH --error={log_dir}/%A_%a_err.txt\n'
                f'#SBATCH --array=0-{n_tasks - 1}%{MAX_RUNNING}\n'
                '#SBATCH --ntasks=1\n'
                '#SBATCH --cpus-per-task=1\n'
                f'#SBATCH --mem={MEM_GB}G\n'
                f'#SBATCH --time={TIME}\n'
                f'#SBATCH --account={ACCOUNT}\n\n'
                'export PYTHONUNBUFFERED=1\n'
                'export OMP_NUM_THREADS=1\n'
                f'export ML_CLASSIFY_TASKS={task_file}\n'
                f'{sys.executable} {os.path.abspath(__file__)}\n')
    result = subprocess.run(['sbatch', script], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'sbatch failed: {result.stderr.strip()}')
    print(f'{len(tasks)} cuts in {n_tasks} array tasks ({CUTS_PER_TASK} cuts each): {result.stdout.strip()}')
    print(f'Logs: {log_dir}   (progress: squeue -u $USER). Run this script again when they have finished.')


def collect():
    """One predictions table per sector, and a summary against the pipeline's tags."""
    for sector in SECTORS:
        files = [pred_path(*c) for c in all_cuts() if c[0] == sector and os.path.exists(pred_path(*c))]
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
        out = f'{OUT_DIR}/S{sector}_ml_predictions.csv.gz'
        df.to_csv(out, index=False, compression='gzip')
        print(f'\n===== Sector {sector}: {len(df)} frame-bin {FRAME_BINS} events in {len(files)} cuts -> {out}')
        if not len(df):
            continue
        print(df.pred_class.value_counts().to_string())
        print(f'review (top p < {REVIEW_MAX_P}): {df.review.mean():.1%};  '
              f'median domain_out_frac {df.domain_out_frac.median():.3f}')
        tag = df.classification.astype(str)
        tag = pd.Series(np.where(tag.str.startswith('V'), 'catalogue variable', np.where(tag == '-', 'untagged', tag)),
                        name='pipeline tag')
        print('\nPipeline tag (rows) against the model (columns), fraction of each row:')
        table = pd.crosstab(tag, df.pred_class, normalize='index').round(3)
        table.insert(0, 'n', tag.value_counts())
        print(table.to_string())


def main():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        run_task()
        return
    if not os.path.exists(MODEL):
        sys.exit(f'No model at {MODEL}')
    load_model()                              # fail on a version mismatch before submitting anything
    todo = todo_cuts()
    if not todo:
        if COLLECT:
            collect()
        else:
            print('All cuts done (COLLECT = False, so no combined table). Next: ml_sector_sample.py')
    elif ARRAY:
        submit_array(todo)
    elif SLURM and 'SLURM_JOB_ID' not in os.environ:
        submit_job(todo)
    else:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        run_here(todo)


if __name__ == '__main__':
    main()

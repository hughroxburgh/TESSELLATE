"""
How fast is the ML feature extraction for different numbers of workers? Run on the cluster.

Edit the CONFIG block, then:  python ml_cpu_benchmark.py

SLURM = True submits this file as ONE job with JOB_CPUS CPUs (log in OUT_DIR); SLURM = False runs it right here.
In the job it times the per-event ML features (the parallel part of classify_cut) on N_EVENTS frame-bin-1
events of one cut, for each worker count in WORKERS:
  numbers           that many joblib workers
  'cpu_count'       multiprocessing.cpu_count() -- what the Detector uses now (every core on the NODE)
  'allocated'       len(os.sched_getaffinity(0)) -- the cores SLURM actually gave the job
and each mode in MODES:
  per event         one task per event (what extract_cut_features does now)
  chunked           CHUNKS_PER_WORKER chunks of events per worker (fewer, bigger tasks)
  chunked + RAM     chunked, with the flux cube read into memory first; joblib then shares it with the workers
                    through a temporary file in /dev/shm (memory) instead of each worker memory-mapping the cube
                    on Lustre
(1 worker runs 'per event' and, if asked for, '+ RAM', in this process.) Every run starts fresh workers, as each
cut's detector job does, so the start-up time is included.

Prints a table: workers, mode, start-up s, total s, events/s, speed-up over 1 worker, CPU efficiency (CPU time
used / (wall time x workers) -- low = workers waiting, not computing), sys_share (the part of that CPU time spent
in the operating system -- high = the kernel, e.g. file-system locking, not our code) and a check that the
features are identical to the first run's.
"""

import copy
import os
import shutil
import subprocess
import sys
import time
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTOR, CAM, CCD, CUT = 55, 1, 1, 30
N_EVENTS = 3000             # frame-bin-1 events timed (a fixed random choice); 1 worker takes ~20-30 ms per event
WORKERS = [1, 8, 16, 32]    # can also hold 'allocated' and 'cpu_count'
MODES = ['per event', 'chunked', 'chunked + RAM']
CHUNKS_PER_WORKER = 4       # chunked modes: the events split into workers x this many tasks
SEED = 0

SLURM = True                # True = submit one job; False = run right here
JOB_CPUS = 32               # CPUs requested for the job (as for a search job)
MEM_PER_CPU_GB = 1
JOB_TIME = '00:45:00'
ACCOUNT = 'oz335'
OUT_DIR = '/fred/oz335/hroxburg/dev/ml_cpu_benchmark'
# ----------------


def allocated_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:      # macOS has no sched_getaffinity
        return multiprocessing.cpu_count()


def _features_chunk(evs, cd, cfg):
    """Chunked modes: one task works through a list of events (the cut data is sent once per chunk)."""
    from tessellate.ml_classifier import _event_features_safe
    return [_event_features_safe(ev, cd, cfg) for ev in evs]


def benchmark():
    import resource
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor
    import tessellate.ml_classifier as ml

    print(f'multiprocessing.cpu_count() = {multiprocessing.cpu_count()}, allocated (sched_getaffinity) = '
          f'{allocated_cpus()}, SLURM_CPUS_PER_TASK = {os.environ.get("SLURM_CPUS_PER_TASK")}', flush=True)
    if os.path.isdir('/dev/shm'):
        print(f'/dev/shm free: {shutil.disk_usage("/dev/shm").free / 1e9:.1f} GB '
              f'(JOBLIB_TEMP_FOLDER = {os.environ.get("JOBLIB_TEMP_FOLDER")})', flush=True)

    t = time.time()
    events = ml._as_loaded(ml.load_cut_events(DATA_PATH, SECTOR, CAM, CCD, CUT))
    fb1 = events[events.frame_bin == 1]
    fb1 = fb1.sample(min(N_EVENTS, len(fb1)), random_state=SEED)
    evs = [ev for _, ev in fb1.iterrows()]
    print(f'S{SECTOR} C{CAM} C{CCD} cut {CUT}: {len(events)} events, timing {len(evs)} frame-bin-1 events '
          f'(loaded in {time.time() - t:.1f}s)', flush=True)
    t = time.time()
    cd = ml._CutData(DATA_PATH, SECTOR, CAM, CCD, CUT, 8, frame_stats=True)
    print(f'cut data + frame noise (serial, once per cut): {time.time() - t:.1f}s', flush=True)
    cd_ram = None
    if any('RAM' in m for m in MODES):
        t = time.time()
        cd_ram = copy.copy(cd)
        cd_ram.flux = np.array(cd.flux)                     # the whole cube in memory (a plain array)
        cd_ram._binned = {}
        print(f'flux cube read into memory: {cd_ram.flux.nbytes / 1e9:.2f} GB in {time.time() - t:.1f}s', flush=True)
    print(flush=True)
    cfg = {**ml.DEFAULT_CONFIG, 'crossmatch': False, 'max_tagged': None}

    def cpu_times(who):
        r = resource.getrusage(who)
        return np.array([r.ru_utime, r.ru_stime])

    rows, reference = [], None
    for w in WORKERS:
        n = {'allocated': allocated_cpus(), 'cpu_count': multiprocessing.cpu_count()}.get(w, w)
        modes = [m for m in MODES if n > 1 or m in ('per event', 'chunked + RAM')]
        for mode in modes:
            data = cd_ram if 'RAM' in mode else cd
            label = mode if n > 1 else mode.replace('chunked + ', '+ ')
            get_reusable_executor().shutdown(wait=True)     # fresh workers, like a new detector job
            who = resource.RUSAGE_SELF if n == 1 else resource.RUSAGE_CHILDREN   # children: finished ones only
            cpu0 = cpu_times(who)
            t0 = time.time()
            if n == 1:
                out = [ml._event_features_safe(ev, data, cfg) for ev in evs]
                startup = 0.0
            else:
                Parallel(n_jobs=n)(delayed(abs)(i) for i in range(n))   # just starts the workers
                startup = time.time() - t0
                if mode == 'per event':
                    out = Parallel(n_jobs=n)(delayed(ml._event_features_safe)(ev, data, cfg) for ev in evs)
                else:
                    chunks = np.array_split(np.arange(len(evs)), n * CHUNKS_PER_WORKER)
                    parts = Parallel(n_jobs=n)(delayed(_features_chunk)([evs[i] for i in c], data, cfg)
                                               for c in chunks if len(c))
                    out = [r for p in parts for r in p]
            total = time.time() - t0
            get_reusable_executor().shutdown(wait=True)
            user, system = cpu_times(who) - cpu0
            same = None
            if reference is None:
                reference = pd.DataFrame([r or {} for r in out])
            else:
                same = pd.DataFrame([r or {} for r in out]).equals(reference)
            rows.append(dict(workers=f'{n} ({w})' if isinstance(w, str) else str(n), mode=label,
                             startup_s=round(startup, 1), total_s=round(total, 1),
                             events_per_s=round(len(evs) / total, 1),
                             cpu_efficiency=f'{(user + system) / (total * n):.0%}',
                             sys_share=f'{system / max(user + system, 1e-9):.0%}', identical=same))
            print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.insert(5, 'speedup', (table.loc[0, 'total_s'] / table['total_s']).round(1))
    print('\n' + table.to_string(index=False))


def submit():
    os.makedirs(OUT_DIR, exist_ok=True)
    script = f'{OUT_DIR}/ml_cpu_benchmark.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                '#SBATCH --job-name=ml_cpu_benchmark\n'
                f'#SBATCH --output={OUT_DIR}/%j_out.txt\n'
                f'#SBATCH --error={OUT_DIR}/%j_err.txt\n'
                '#SBATCH --ntasks=1\n'
                f'#SBATCH --cpus-per-task={JOB_CPUS}\n'
                f'#SBATCH --mem-per-cpu={MEM_PER_CPU_GB}G\n'
                f'#SBATCH --time={JOB_TIME}\n'
                f'#SBATCH --account={ACCOUNT}\n\n'
                'export PYTHONUNBUFFERED=1\n'
                f'{sys.executable} {os.path.abspath(__file__)}\n')
    result = subprocess.run(['sbatch', script], capture_output=True, text=True)
    if result.returncode:
        sys.exit(f'sbatch failed: {result.stderr.strip()}')
    print(f'{result.stdout.strip()} -- results in {OUT_DIR}/<jobid>_out.txt')


if __name__ == '__main__':
    if SLURM and 'SLURM_JOB_ID' not in os.environ:
        submit()
    else:
        benchmark()

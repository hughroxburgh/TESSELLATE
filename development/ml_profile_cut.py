"""
Time the ML feature extraction on one real cut, to find out what makes a sector
slow. Run it on the cluster, ideally where the extraction itself runs (an
interactive job, or the login node for a quick look -- it uses one core and a
few minutes).

It times the steps for a random sample of the cut's events twice: reading from
the memory-mapped cube (what ml_extract_features does now) and from a copy of
the cube loaded into memory. A big difference means disk reads are the problem.

Edit the CONFIG block, then:  python ml_profile_cut.py
"""

import cProfile
import io
import os
import pstats
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate import ml_classifier as mc

# ---- CONFIG ----
DATA_PATH = '/fred/oz335/TESSdata'
SECTOR = 55
CAM, CCD, CUT = 1, 1, 20
N_EVENTS = 100              # random events to time (None = every event in the cut -- slow)
N_WORKERS = 16              # for the projected time per sector
SEED = 0
# ----------------


def time_events(sample, cd, cfg, profile=False):
    prof = cProfile.Profile() if profile else None
    times = []
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        if prof:
            prof.enable()
        for _, ev in sample.iterrows():
            t0 = time.time()
            try:
                mc._event_features(ev, cd, cfg)
            except Exception:
                pass
            times.append(time.time() - t0)
        if prof:
            prof.disable()
    return np.array(times), prof


def main():
    cfg = dict(mc.DEFAULT_CONFIG)
    t0 = time.time()
    events = mc.load_cut_events(DATA_PATH, SECTOR, CAM, CCD, CUT)
    print(f'S{SECTOR} C{CAM} C{CCD} cut {CUT}: {len(events)} events')
    print('  by frame_bin:      ', events.frame_bin.value_counts().sort_index().to_dict())
    print('  by classification: ', events.classification.value_counts().to_dict())

    t1 = time.time()
    mc.table_features(events)
    t2 = time.time()
    cd = mc._CutData(DATA_PATH, SECTOR, CAM, CCD, CUT, frame_stats=cfg['frame_stats'])
    t3 = time.time()
    size_gb = cd.flux.size * cd.flux.itemsize / 1e9
    print(f'\nCube {cd.flux.shape} {cd.flux.dtype}, {size_gb:.2f} GB')
    print(f'  read events {t1 - t0:.1f} s   table features {t2 - t1:.1f} s   open cube + frame-noise pass {t3 - t2:.1f} s')

    sample = events if N_EVENTS is None else events.sample(min(N_EVENTS, len(events)), random_state=SEED)
    mm, prof = time_events(sample, cd, cfg, profile=True)
    print(f'\nPer event, memory-mapped cube: mean {mm.mean():.3f} s, median {np.median(mm):.3f} s, '
          f'max {mm.max():.2f} s  (with the profiler on, which adds a little)')
    by_bin = sample.assign(t=mm).groupby('frame_bin').t.agg(['mean', 'size']).round(3)
    print('  by frame_bin:\n  ' + by_bin.to_string().replace('\n', '\n  '))

    t4 = time.time()
    cd.flux = np.load(f'{mc._cut_base(DATA_PATH, SECTOR, CAM, CCD, CUT)}_ReducedFlux.npy')
    t5 = time.time()
    ram, _ = time_events(sample, cd, cfg)
    print(f'\nCube loaded into memory in {t5 - t4:.1f} s; per event then: mean {ram.mean():.3f} s, '
          f'median {np.median(ram):.3f} s')

    per_cut = (t3 - t0) + len(events) * mm.mean()
    print(f'\nProjected: {per_cut / 60:.1f} min for this cut; a sector of 1024 such cuts on {N_WORKERS} workers '
          f'~{1024 * per_cut / N_WORKERS / 3600:.1f} h')

    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats('tottime').print_stats(20)
    print('\nWhere the time goes (memory-mapped run, sorted by own time):')
    print(s.getvalue()[s.getvalue().find('ncalls'):])


if __name__ == '__main__':
    main()

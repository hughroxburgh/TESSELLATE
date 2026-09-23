"""
Synthetic tessellate cuts for testing tessellate/ml_classifier.py end to end
before real data is available: generates fake cuts, extracts features, then
trains and evaluates the classifier exactly as on real data.

Edit the CONFIG block, then:  python ml_synthetic.py

Writes, under OUT (~300 MB for 5 cuts; takes a few minutes):
  TESSdata/Sector{S}/Cam1/Ccd1/Cut{k}of64/   ReducedFlux.npy, Times.npy, detected_events.csv
  sorted/{Group}/                            manual_sort.py-style output (PNG names + events.csv)
  truth.csv                                  true class of every synthetic event
  features.csv                               from ml_classifier.build_feature_table
  eval/                                      ml_train_eval.py outputs, incl. truth_check.txt

The injected classes are caricatures -- flares, moving asteroids, cosmic-ray
hits, periodic variables, and several kinds of junk (flickering pixels,
subtraction dipoles, scattered-light glints, momentum-dump frames). Good
results here only show that the plumbing works and that the features respond
to the physics they target; they say nothing about performance on real data.
"""

import os
import sys

import numpy as np
import pandas as pd

# ---- CONFIG ----
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic')   # git-ignored
N_CUTS = 5
SECTOR = 99                 # fake sector number
SIZE = 80                   # cut size in pixels (real cuts are ~266)
FRAMES_PER_ORBIT = 1200
LABEL_FRAC = 0.7            # fraction of untagged frame-bin-1 events that get "manually sorted"
SEED = 0                    # same seed = same data

GENERATE = True             # False = reuse the cuts already in OUT
EVALUATE = True             # extract features, train and evaluate
ABLATION = True             # also score subsets of feature groups (slower)
IMPORTANCE = True
# ----------------

CADENCE = 600 / 86400
PSF_SIGMA = 0.75
NOISE = 5.0


def add_psf(flux, frames, xs, ys, amps, half=4):
    """Add a Gaussian PSF of total flux amps[i] at (xs[i], ys[i]) in frames[i]."""
    _, h, w = flux.shape
    for f, x, y, a in zip(np.atleast_1d(frames), np.broadcast_to(xs, np.shape(frames)),
                          np.broadcast_to(ys, np.shape(frames)), np.broadcast_to(amps, np.shape(frames))):
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(yi - half, 0), min(yi + half + 1, h), max(xi - half, 0), min(xi + half + 1, w)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        k = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * PSF_SIGMA ** 2)) / (2 * np.pi * PSF_SIGMA ** 2)
        flux[f, y0:y1, x0:x1] += a * k


def add_static(flux, x, y, amp_series, half=4):
    """A fixed-position source whose flux varies as amp_series (one value per frame)."""
    _, h, w = flux.shape
    xi, yi = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(yi - half, 0), min(yi + half + 1, h), max(xi - half, 0), min(xi + half + 1, w)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    k = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * PSF_SIGMA ** 2)) / (2 * np.pi * PSF_SIGMA ** 2)
    flux[:, y0:y1, x0:x1] += amp_series[:, None, None].astype(np.float32) * k


def place(rng, taken, size, margin=5, min_sep=7):
    """Random position at least min_sep from earlier objects (relaxed if the cut is crowded)."""
    while min_sep > 2:
        for _ in range(500):
            x, y = rng.uniform(margin, size - margin - 1, 2)
            if all(np.hypot(x - a, y - b) >= min_sep for a, b in taken):
                taken.append((x, y))
                return x, y
        min_sep -= 1
    raise RuntimeError('Could not place object; increase --size')


def detect(flux, x, y, near, window=3):
    """Pipeline-like event boundaries from the 3x3 light curve around frame `near`."""
    xi, yi = int(round(x)), int(round(y))
    lc = np.nansum(flux[:, yi - 1:yi + 2, xi - 1:xi + 2], axis=(1, 2))
    lc[np.all(~np.isfinite(flux[:, yi - 1:yi + 2, xi - 1:xi + 2]), axis=(1, 2))] = np.nan
    # significance against the image noise (as the difference-image detection is), not the light
    # curve's own scatter -- which for a variable star is dominated by the variability itself
    baseline = pd.Series(lc).rolling(301, center=True, min_periods=50).median().to_numpy()
    z = (lc - baseline) / (3 * NOISE)
    lo, hi = max(near - window, 0), min(near + window + 1, len(z))
    if not np.isfinite(z[lo:hi]).any():
        return None
    p = lo + int(np.nanargmax(z[lo:hi]))
    if not z[p] >= 5:
        return None
    s = e = p
    while s - 1 >= 0 and z[s - 1] >= 3:
        s -= 1
    while e + 1 < len(z) and z[e + 1] >= 3:
        e += 1
    ze = z[s:e + 1]
    return dict(xint=xi, yint=yi, frame_start=s, frame_end=e, frame_max=s + int(np.nanargmax(ze)),
                lc_sig_max=float(np.nanmax(ze)), lc_sig_med=float(np.nanmean(ze)),
                n_detections=int(np.sum(ze >= 3)), flux_max=float(lc[s + int(np.nanargmax(ze))]))


def make_cut(rng, size, n_orbit):
    t_first = 60000 + np.arange(n_orbit) * CADENCE
    time = np.concatenate([t_first, t_first[-1] + 1.0 + np.arange(n_orbit) * CADENCE])   # 1-day downlink gap
    n = len(time)
    flux = rng.normal(0, NOISE, (n, size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    # slow scattered-light ramp at the start of each orbit (the detrending should absorb it)
    for start in (0, n_orbit):
        ramp = 12 * np.exp(-np.arange(200) / 40)
        flux[start:start + 200] += (ramp[:, None, None] * (xx / size)[None]).astype(np.float32)

    taken, objects = [], []

    # -- Flare stars: fast rise, exponential decay; some with rotational modulation -- #
    for _ in range(10):
        x, y = place(rng, taken, size)
        if rng.random() < 0.5:
            add_static(flux, x, y, rng.uniform(5, 20) * np.sin(2 * np.pi * time / rng.uniform(0.5, 3)))
        peaks = []
        for _ in range(rng.choice([1, 1, 2, 3])):
            t0 = int(rng.integers(40, n - 80))
            A, tr, td = np.exp(rng.uniform(np.log(80), np.log(800))), rng.uniform(0.3, 1.2), rng.uniform(1.5, 12)
            f = np.arange(t0 - 6, t0 + 70)
            prof = np.where(f < t0, np.exp((f - t0) / tr), np.exp(-(f - t0) / td)) * A
            add_psf(flux, f, x, y, prof)
            peaks.append(t0)
        objects.append(dict(cls='Flare', x=x, y=y, peaks=peaks))

    # -- Asteroids: a PSF moving in a straight line past the event pixel -- #
    for _ in range(8):
        x, y = place(rng, taken, size)
        fc = int(rng.integers(80, n - 80))
        speed, ang = rng.uniform(0.08, 0.6), rng.uniform(0, 2 * np.pi)
        f = np.arange(fc - int(8 / speed), fc + int(8 / speed) + 1)
        f = f[(f >= 0) & (f < n)]
        add_psf(flux, f, x + speed * np.cos(ang) * (f - fc), y + speed * np.sin(ang) * (f - fc),
                np.exp(rng.uniform(np.log(120), np.log(500))))
        objects.append(dict(cls='Asteroid', x=x, y=y, peaks=[fc], pipeline='Asteroid' if speed > 0.35 else '-'))

    # -- Cosmic rays: one frame, one or two pixels -- #
    for _ in range(12):
        x, y = place(rng, taken, size, min_sep=4)
        f, xi, yi = int(rng.integers(5, n - 5)), int(round(x)), int(round(y))
        E = np.exp(rng.uniform(np.log(150), np.log(1500)))
        flux[f, yi, xi] += E
        if rng.random() < 0.25:
            flux[f, yi, xi + 1] += 0.3 * E
        objects.append(dict(cls='CosmicRay', x=xi, y=yi, peaks=[f], window=0,
                            pipeline='CosmicRay' if rng.random() < 0.6 else '-'))

    # -- Periodic variables: sinusoids and RR Lyrae-like sawtooths; events are cycle maxima -- #
    for _ in range(8):
        x, y = place(rng, taken, size)
        P, A, phi0 = np.exp(rng.uniform(np.log(0.08), np.log(0.8))), rng.uniform(60, 200), rng.uniform(0, 1)
        phase = ((time - time[0]) / P + phi0) % 1
        if rng.random() < 0.5:
            wave = np.sin(2 * np.pi * phase)
        else:
            wave = np.where(phase < 0.15, phase / 0.15, 1 - (phase - 0.15) / 0.85) * 2 - 1
        add_static(flux, x, y, A * wave)
        maxima = np.flatnonzero((wave[1:-1] >= wave[:-2]) & (wave[1:-1] > wave[2:]) & (wave[1:-1] > 0.9)) + 1
        maxima = maxima[(maxima > 20) & (maxima < n - 20)]
        objects.append(dict(cls='Variable', x=x, y=y, peaks=list(rng.choice(maxima, 2, replace=False)), window=4))

    # -- Junk -- #
    for _ in range(4):      # flickering pixel with heavy-tailed noise
        x, y = place(rng, taken, size, min_sep=5)
        xi, yi = int(round(x)), int(round(y))
        flux[:, yi, xi] += (rng.standard_t(1.5, n) * 6).astype(np.float32)
        order = np.argsort(-np.nan_to_num(flux[:, yi, xi]))
        objects.append(dict(cls='Junk', x=xi, y=yi, peaks=list(order[:2]), window=0, kind='flicker'))
    for _ in range(3):      # subtraction dipole from misregistration of a bright star
        x, y = place(rng, taken, size)
        f0, k, A = int(rng.integers(20, n - 20)), int(rng.integers(2, 7)), rng.uniform(300, 1500)
        frames = np.arange(f0, f0 + k)
        add_psf(flux, frames, x + 0.35, y, A)
        add_psf(flux, frames, x - 0.35, y, -A)
        objects.append(dict(cls='Junk', x=x + 0.6, y=y, peaks=[f0 + k // 2], kind='dipole'))
    for start in (0, n_orbit):   # scattered-light glint early in an orbit
        for _ in range(2):
            x, y = place(rng, taken, size, min_sep=9)
            fc, width, s = start + int(rng.integers(10, 100)), rng.uniform(2, 6), rng.uniform(2.5, 4)
            amp = rng.uniform(25, 50) * np.exp(-0.5 * ((np.arange(n) - fc) / width) ** 2)
            blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s ** 2))
            flux += (amp[:, None, None] * blob[None]).astype(np.float32)
            objects.append(dict(cls='Junk', x=x, y=y, peaks=[fc], kind='glint'))
    for _ in range(2):      # momentum-dump frame: correlated structure everywhere at once
        from scipy.ndimage import gaussian_filter
        f = int(rng.integers(20, n - 20))
        field = gaussian_filter(rng.normal(0, 1, (size, size)), 1.5)
        field *= 110 / np.abs(field).max()
        flux[f] += field.astype(np.float32)
        spots = np.dstack(np.unravel_index(np.argsort(-field.ravel())[:40], field.shape))[0]
        chosen = []
        for yi, xi in spots:
            if 3 <= xi < size - 3 and 3 <= yi < size - 3 and all(np.hypot(xi - a, yi - b) > 5 for a, b in chosen):
                chosen.append((xi, yi))
            if len(chosen) == 3:
                break
        for xi, yi in chosen:
            objects.append(dict(cls='Junk', x=xi, y=yi, peaks=[f], window=0, kind='dump'))

    for f in rng.choice(n, 4, replace=False):   # momentum-dump gaps
        flux[f] = np.nan
    return time, flux, objects


def to_binned(i, fb, n_orbit):
    n_first = -(-n_orbit // fb)
    return i // fb if i < n_orbit else n_first + (i - n_orbit) // fb


def events_table(time, flux, objects, sector, cam, ccd, cut, n_orbit, rng):
    rows, objid = [], 0
    for obj in objects:
        objid += 1
        found = []
        for p in obj['peaks']:
            ev = detect(flux, obj['x'], obj['y'], int(p), obj.get('window', 3))
            if ev is None or any(ev['frame_start'] <= e['frame_end'] and ev['frame_end'] >= e['frame_start'] for e in found):
                continue
            found.append(ev)
        for eventid, ev in enumerate(found, 1):
            dur = ev['frame_end'] - ev['frame_start'] + 1
            pipeline = obj.get('pipeline', '-')
            if obj.get('kind') == 'flicker' and dur == 1 and rng.random() < 0.5:
                pipeline = 'Junk'
            if obj['cls'] == 'CosmicRay' and dur > 1:
                pipeline = '-'
            x = obj['x'] + rng.normal(0, 0.1)
            y = obj['y'] + rng.normal(0, 0.1)
            row = dict(objid=objid, eventid=eventid, classification=pipeline, sector=sector, camera=cam, ccd=ccd,
                       cut=cut, xcentroid=x, ycentroid=y, xcentroid_err=0.12, ycentroid_err=0.12,
                       xccd=ev['xint'] + 500, yccd=ev['yint'] + 500, xcentroid_det=x, ycentroid_det=y,
                       xcentroid_psf=x, ycentroid_psf=y, frame_duration=dur, frame_bin=1, flux_sign=1,
                       image_sig_max=ev['lc_sig_max'], lc_flat=1, psf_stacked=0, bad_frame_flag=0, source_mask=0,
                       gaia_id='-', crossbin_ids='[]', asteroid_id=-1, total_events=len(found),
                       true_class=obj['cls'], kind=obj.get('kind', obj['cls']),
                       **{k: v for k, v in ev.items() if k not in ('xint', 'yint')}, xint=ev['xint'], yint=ev['yint'])
            for k in ('start', 'end', 'max'):
                row[f'mjd_{k}'] = time[ev[f'frame_{k}']]
            row['mjd_duration'] = row['mjd_end'] - row['mjd_start']
            rows.append(row)

            # a coarser-binned detection of longer events, as the multi-bin search produces,
            # linked to the fine-binned one through a shared crossbin id
            if dur >= 6:
                fb, cid = 3, len(rows)
                row['crossbin_ids'] = f'[{cid}]'
                b = {k: to_binned(ev[f'frame_{k}'], fb, n_orbit) for k in ('start', 'end', 'max')}
                rows.append({**row, 'objid': objid + 10000, 'eventid': eventid, 'frame_bin': fb, 'classification': '-',
                             'frame_start': b['start'], 'frame_end': b['end'], 'frame_max': b['max'],
                             'frame_duration': b['end'] - b['start'] + 1})
    df = pd.DataFrame(rows)
    for c in ['snr_psf', 'psf_like', 'psf_diff', 'ellipticity', 'fwhm', 'neg_extent', 'com_motion',
              'gaussian_score', 'known_asteroid_dist_px', 'gal_b']:
        df[c] = np.nan      # not simulated: leave these to the recomputed features
    return df


def generate():
    rng = np.random.default_rng(SEED)
    truth = []
    for cut in range(1, N_CUTS + 1):
        time, flux, objects = make_cut(rng, SIZE, FRAMES_PER_ORBIT)
        path = f'{OUT}/TESSdata/Sector{SECTOR}/Cam1/Ccd1/Cut{cut}of64'
        os.makedirs(path, exist_ok=True)
        base = f'{path}/sector{SECTOR}_cam1_ccd1_cut{cut}_of64'
        np.save(f'{base}_ReducedFlux.npy', flux)
        np.save(f'{base}_Times.npy', time)
        events = events_table(time, flux, objects, SECTOR, 1, 1, cut, FRAMES_PER_ORBIT, rng)
        events.drop(columns=['true_class', 'kind']).to_csv(f'{path}/detected_events.csv', index=False)
        truth.append(events)
        print(f'cut {cut}: {len(events)} events  {events.true_class.value_counts().to_dict()}')

    truth = pd.concat(truth, ignore_index=True)
    truth[['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid', 'frame_bin', 'true_class', 'kind',
           'classification']].to_csv(f'{OUT}/truth.csv', index=False)

    # -- manual_sort.py-style output for a subset of the events the filters would have shown -- #
    sortable = truth[(truth.classification == '-') & (truth.frame_bin == 1)]
    sorted_events = sortable[rng.random(len(sortable)) < LABEL_FRAC]
    for group, rows in sorted_events.groupby('true_class'):
        gdir = f'{OUT}/sorted/{group}'
        os.makedirs(gdir, exist_ok=True)
        rows.drop(columns=['true_class', 'kind']).to_csv(f'{gdir}/events.csv', index=False)
        for r in rows.itertuples():
            open(f'{gdir}/S{r.sector}C{r.camera}C{r.ccd}C{r.cut}O{r.objid}E{r.eventid}.png', 'w').close()
    print(f'\n{len(sorted_events)} events "manually sorted" into {OUT}/sorted: '
          f'{sorted_events.true_class.value_counts().to_dict()}')
    print(f'pipeline weak labels: {truth[truth.classification != "-"].classification.value_counts().to_dict()}')


def evaluate_synthetic():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from tessellate.ml_classifier import build_feature_table
    from ml_train_eval import run

    features = build_feature_table(f'{OUT}/TESSdata', SECTOR, cams=[1], ccds=[1], cuts=range(1, N_CUTS + 1),
                                   config={'crossmatch': False})
    features.to_csv(f'{OUT}/features.csv', index=False)
    run([f'{OUT}/features.csv'], f'{OUT}/sorted', f'{OUT}/eval', ablation=ABLATION, importance=IMPORTANCE,
        review=50, truth=f'{OUT}/truth.csv')
    print(f'\nResults in {OUT}/eval (report.txt, truth_check.txt, plots)')


if __name__ == '__main__':
    if GENERATE:
        if os.path.exists(f'{OUT}/sorted'):
            import shutil
            shutil.rmtree(f'{OUT}/sorted')     # old labels would mix with the new cuts
        generate()
    if EVALUATE:
        evaluate_synthetic()

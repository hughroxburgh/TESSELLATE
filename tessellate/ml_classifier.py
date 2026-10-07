"""
Machine-learning classification of tessellate events.

Classifies detected events (rows of detected_events.csv) into the manual-sort
taxonomy -- Junk, CosmicRay, Systematic, Blend, Asteroid, Flare, Variable,
Interesting -- from several kinds of information, each a feature group with
its own prefix:

  tab_    statistics the pipeline already stores in the event table
  lc_     the event's own light-curve shape (duration-normalised)
  shape_  the light curve through the event resampled onto a fixed grid
  ctx_    how the event sits within the full-sector light curve at that
          position: recurrence, periodicity, noise character, orbit edges
  pix_    the pixels around the event in the difference-imaged flux cube
  xm_     Gaia / variable-catalogue context, recomputed for every event
          (extracted, but left out of the model by default: see HOST_FEATURES)

The light curve is the same 3x3 box sum (tools.Generate_LC) shown in the
Navigator plots used for manual sorting. Features are extracted per cut from
the reduced flux cube, so extraction can run wherever the data lives (e.g. on
the cluster) and only the small feature tables need to be moved.

Workflow
--------
    from tessellate.ml_classifier import (build_feature_table, load_manual_labels,
                                           attach_labels, EventClassifier, evaluate)

    features = build_feature_table(data_path, sector=55)
    labels = attach_labels(features, load_manual_labels('/path/to/sorted/images'))

    clf = EventClassifier().fit(features, labels)   # grouped CV, calibration, final fit
    report = evaluate(clf.oof_)                      # out-of-fold metrics on manual labels
    predictions = clf.predict(features)
    clf.save('event_classifier.joblib')

Labels
------
Label an event by the physical event that defines its record (its peak,
significance, duration and position); anything else in the light curve is
background. Classes:

  Junk         artefacts that don't fit the classes below
  CosmicRay    one-frame hits, including those whose event window got stretched by noise
  Systematic   real-looking, but one of many events at the same time (pointing jitter,
               scattered light, unstable stretches)
  Blend        the record's measurements come from different sources (e.g. a cosmic
               ray on an asteroid track), so it can't be trusted as one object
  Asteroid, Flare, Variable
               Flare means flare-shaped, whether or not a star is there: a flare in
               empty sky (e.g. a GRB afterglow) is a Flare too. Whether it has a host
               is decided afterwards from the localisation and the Gaia crossmatch, so
               the features that encode it (HOST_FEATURES) are left out of the model
               by default. A flare on a variable star is a Flare -- the variability is
               its baseline.
  Interesting  real, but none of the above

Junk, CosmicRay, Systematic and Blend count as artefacts. Sort folders can use
other names and be mapped with load_manual_labels(rename=...).

Manual labels (development/manual_sort.py) are the ground truth, and the only
labels used for evaluation and calibration. The pipeline's own Junk / CosmicRay /
Asteroid tags can be added as down-weighted weak labels: they cover parts of
feature space the manual sort never sees (the sort only shows events that
already passed filter_events), but they are deterministic functions of a few
features, so on their own they only teach the model to reproduce the rules.

Columns the pipeline fills only for events the rules did not already tag
(com_motion, gaussian_score, the gaia_id / nearest_gaia_* crossmatch, the
variable-catalogue classification, asteroid_id) are never used as features --
their missingness encodes the rule-based label. The reverse case, filled only
for tagged events (known_asteroid_dist_px), is extracted but always left out
of the model (LEAKY_FEATURES). The equivalent information is
recomputed for every event (pix_cen_*, lc_fit_*, xm_*).
"""

import os
import re
import warnings

import numpy as np
import pandas as pd


# -- Taxonomy -- #
CLASSES = ['Junk', 'CosmicRay', 'Systematic', 'Blend', 'Asteroid', 'Flare', 'Variable', 'Interesting']
ARTEFACT_CLASSES = ['Junk', 'CosmicRay', 'Systematic', 'Blend']   # see "Labels" in the module docstring
PIPELINE_CLASSES = ['Junk', 'CosmicRay', 'Asteroid']

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
META_COLS = KEY_COLS + ['frame_bin', 'flux_sign', 'classification', 'xcentroid', 'ycentroid', 'mjd_max',
                        'crossbin_ids']
FEATURE_GROUPS = ('tab', 'lc', 'shape', 'ctx', 'pix', 'xm')

# -- The classifier shipped with the package and used by the detector -- #
ML_VERSION = '1.0.0'        # = development model v5_E (labels to 2026-10-06, incl. S54); see development/claude_context.md
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rfc_files',
                                  f'event_classifier_v{ML_VERSION}.joblib')
ML_MAX_JOBS = 8             # workers for the per-event features: more is slower on the cluster (ml_cpu_benchmark.py:
                            # 8 workers 173 ev/s, 16: 164, 32: 109 -- the workers contend for the memory-mapped cube)
CHUNKS_PER_JOB = 4          # each worker gets ~this many chunks of events (one task per event is ~3x slower)
UNSURE_BELOW = 0.6          # most likely class below this -> 'Unsure' (S54: 3% of events at lc_sig_max >= 5,
                            # right only half the time; the rest 95% right)
FEATURE_VERSION = 5     # 5: event margin for the baseline capped at max_mask_pad_days (long events kept their
                        # light-curve features blank before), tab_chain_* (fast-asteroid detection chains)
                        # 4: image at before / peak / after and first / second half (pix_ep_*), gaps in the event
                        # window (lc_gap_frames, lc_spans_gap, lc_event_nan_frac), hour-scale and ring-relative
                        # variability (ctx_red_noise, ctx_ring_corr, ctx_core_ring_ratio), ctx_peak_over_p99

# Features (name prefixes) that say whether a star is at the event position: the Gaia / variable-catalogue
# distances and bit 0 of the reduction's source mask (pixel on a catalogue star). Flare is defined by shape,
# star or no star, so these are left out of the model by default -- nearly every sorted flare is on a star,
# and with them the model would learn to down-rank a flare in empty sky. Whether a star is there is decided
# afterwards from the localisation and the crossmatch.
HOST_FEATURES = ('xm_', 'tab_source_mask_b0')

# Features filled only for events the pipeline already tagged, so their presence gives the tag away (see the
# leakage note in the module docstring). Always left out of the model; still extracted.
#   tab_known_asteroid_dist_px: only set when an event matched a known MPC asteroid, which also tags it Asteroid
LEAKY_FEATURES = ('tab_known_asteroid_dist_px',)

DEFAULT_CONFIG = {
    'detrend_days': 1.0,          # running-median window; keeps events up to a few hours intact
    'max_mask_pad_days': 0.5,     # margin hidden either side of an event for the baseline: min(duration, this)
    'ls_min_period_days': 0.02,
    'ls_max_period_days': 2.0,    # longer periods are unreliable after the reduction's trend removal
    'n_shape': 16,                # points in the duration-normalised shape vector
    'frame_stats': True,          # per-frame noise of the whole cut (one pass over the cube)
    'crossmatch': True,           # Gaia / variable-catalogue context through CutWCS
    'max_tagged': None,           # per cut, at most this many events of each pipeline tag (Junk / CosmicRay /
                                  # Asteroid) get features; they're only weak labels. None = all of them
}

_IMAGE_NAME = re.compile(r'^S(\d+)C(\d+)C(\d+)C(\d+)O(\d+)E(\d+)\.png$', re.IGNORECASE)
_WARNED = set()


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, stacklevel=3)


# ----------------------------- Data access ----------------------------- #

def _cut_path(data_path, sector, cam, ccd, cut, n=8):
    return f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}'


def _cut_base(data_path, sector, cam, ccd, cut, n=8):
    return f'{_cut_path(data_path, sector, cam, ccd, cut, n)}/sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'


def _as_loaded(events):
    """An in-memory event table as load_cut_events would return it: fresh 0..n-1 index, crossbin_ids as lists."""
    import ast

    events = events.reset_index(drop=True).copy()
    if 'crossbin_ids' in events:
        events['crossbin_ids'] = events['crossbin_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return events


def load_cut_events(data_path, sector, cam, ccd, cut, n=8):
    """
    Read a cut's detected_events.csv (same file Navigator.gather_results reads,
    without needing the WCS).
    """
    import ast
    from .tools import load_table

    events = load_table(f'{_cut_path(data_path, sector, cam, ccd, cut, n)}/detected_events.csv')
    if 'crossbin_ids' in events:
        events['crossbin_ids'] = events['crossbin_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return events


def _segment_break(sector, cam, time):
    """
    Downlink break index used by tools.Frame_Bin. Falls back to the largest
    time gap when the TESS vectors file is not available (e.g. off-cluster).
    """
    try:
        from .tools import Get_Tess_Downlink
        return int(Get_Tess_Downlink(sector, cam, time))
    except Exception:
        _warn_once(('vectors', sector, cam),
                   f'TESS vectors unavailable for sector {sector} camera {cam}; using the largest time gap '
                   'as the downlink break. Copy TessVectors_S*_C*_FFI.csv locally to match Frame_Bin exactly.')
        return int(np.nanargmax(np.diff(time)) + 1)


def _bin_frames(arr, frame_bin, break_idx):
    """
    Mean over consecutive frame_bin frames, separately either side of the
    downlink break -- the same binning as tools.Frame_Bin, for any array whose
    first axis is time.
    """
    arr = np.asarray(arr, dtype=np.float32 if np.ndim(arr) > 1 else float)
    if frame_bin <= 1:
        return arr

    def _segment(a):
        m = -(-len(a) // frame_bin)
        pad = m * frame_bin - len(a)
        if pad:
            a = np.concatenate([a, np.full((pad,) + a.shape[1:], np.nan, dtype=a.dtype)])
        return np.nanmean(a.reshape((m, frame_bin) + a.shape[1:]), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.concatenate([_segment(arr[:break_idx]), _segment(arr[break_idx:])])


def _raw_range(index, frame_bin, break_idx, n_raw):
    """Raw frame range [lo, hi) averaged into binned frame `index`."""
    if frame_bin <= 1:
        return index, index + 1
    n_first = -(-break_idx // frame_bin)
    if index < n_first:
        lo = index * frame_bin
        return lo, min(lo + frame_bin, break_idx)
    lo = break_idx + (index - n_first) * frame_bin
    return lo, min(lo + frame_bin, n_raw)


def _stamp(flux, x, y, half):
    """(time, 2*half+1, 2*half+1) cutout centred on (x, y), NaN-padded at the edges."""
    n, h, w = flux.shape
    size = 2 * half + 1
    out = np.full((n, size, size), np.nan, dtype=np.float32)
    y1, x1 = y - half, x - half
    yy1, yy2 = max(0, y1), min(h, y1 + size)
    xx1, xx2 = max(0, x1), min(w, x1 + size)
    if yy1 < yy2 and xx1 < xx2:
        out[:, yy1 - y1:yy2 - y1, xx1 - x1:xx2 - x1] = flux[:, yy1:yy2, xx1:xx2]
    return out


def _frame_noise(flux, chunk=128):
    """Robust (IQR) noise of every frame across the whole cut."""
    noise = np.full(len(flux), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(0, len(flux), chunk):
            block = np.asarray(flux[i:i + chunk], dtype=np.float32).reshape(min(chunk, len(flux) - i), -1)
            q25, q75 = np.nanpercentile(block, [25, 75], axis=1)
            noise[i:i + chunk] = (q75 - q25) / 1.349
    return noise


def _segment_bounds(time, gap_factor=10):
    """Index ranges of continuous stretches of the (binned) time series."""
    dt = np.diff(time)
    cadence = np.nanmedian(dt)
    breaks = np.where(dt > gap_factor * cadence)[0] + 1
    edges = np.concatenate([[0], breaks, [len(time)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class _CutData():
    """Time and (memory-mapped) flux for one cut, with Frame_Bin-consistent stamp access."""

    def __init__(self, data_path, sector, cam, ccd, cut, n=8, frame_stats=True):
        base = _cut_base(data_path, sector, cam, ccd, cut, n)
        self.time = np.load(f'{base}_Times.npy')
        self.flux = np.load(f'{base}_ReducedFlux.npy', mmap_mode='r')
        self.break_idx = _segment_break(sector, cam, self.time)
        self.noise = _frame_noise(self.flux) if frame_stats else None
        self._binned = {}

    def binned(self, frame_bin):
        """(time, per-frame noise, continuous segment bounds) at this binning."""
        if frame_bin not in self._binned:
            t = _bin_frames(self.time, frame_bin, self.break_idx)
            noise = None if self.noise is None else _bin_frames(self.noise, frame_bin, self.break_idx)
            self._binned[frame_bin] = (t, noise, _segment_bounds(t))
        return self._binned[frame_bin]

    def stamp(self, x, y, half, frame_bin):
        return _bin_frames(_stamp(self.flux, x, y, half), frame_bin, self.break_idx)

    def image(self, x, y, half, frame_bin, index):
        """A single binned frame, without binning the whole time series."""
        lo, hi = _raw_range(index, frame_bin, self.break_idx, len(self.time))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(_stamp(self.flux[lo:hi], x, y, half), axis=0)


# ----------------------------- Table features ----------------------------- #

def _n_crossbin(ids):
    if isinstance(ids, str):
        return ids.count(',') + 1 if ids.strip('[] ') else 0
    try:
        return len(ids)
    except TypeError:
        return np.nan


def _time_crowding(events, windows_hr=(3, 12), sep_px=3.0):
    """
    How crowded in time the rest of the cut is around each event -- the sign of
    systematics that make real-looking events appear everywhere at once. Counts
    other events in the same cut and frame bin, more than sep_px away (so not
    this source or its detections at other bins):
      tab_n_simultaneous     peaking within one frame of this event
      tab_n_local_{w}h       peaking within +/- w hours
      tab_rate_ratio_{w}h    that count relative to the cut's average rate; catches
                             unstable stretches where no single frame stands out
    Only this cut's events are used, so each cut can be classified on its own.
    """
    cols = ['tab_n_simultaneous'] + [f'tab_{p}_{w}h' for w in windows_hr for p in ('n_local', 'rate_ratio')]
    out = pd.DataFrame(np.nan, index=events.index, columns=cols)
    if not {'frame_max', 'frame_bin', 'mjd_max', 'xcentroid', 'ycentroid'} <= set(events):
        return out

    keys = [k for k in ['sector', 'camera', 'ccd', 'cut', 'frame_bin'] if k in events]
    widest = max(windows_hr) / 24
    for idx in events.groupby(keys, sort=False).groups.values():
        g = events.loc[idx]
        t = g['mjd_max'].to_numpy(float)
        frame = g['frame_max'].to_numpy(float)
        x, y = g['xcentroid'].to_numpy(float), g['ycentroid'].to_numpy(float)
        order = np.argsort(t)
        ts = t[order]
        finite = np.isfinite(t)
        span = np.ptp(t[finite]) if finite.sum() > 1 else 0.0

        res = np.full((len(g), len(cols)), np.nan)
        for i in np.flatnonzero(finite):
            near = order[np.searchsorted(ts, t[i] - widest, 'left'):np.searchsorted(ts, t[i] + widest, 'right')]
            near = near[np.hypot(x[near] - x[i], y[near] - y[i]) > sep_px]
            res[i, 0] = np.sum(np.abs(frame[near] - frame[i]) <= 1)
            for k, w in enumerate(windows_hr):
                n = np.sum(np.abs(t[near] - t[i]) <= w / 24)
                expected = (len(g) - 1) * min(2 * w / 24 / span, 1) if span > 0 else 0
                res[i, 1 + 2 * k] = n
                res[i, 2 + 2 * k] = n / expected if expected > 0 else np.nan
        out.loc[idx] = res
    return out


def _chain_features(events, max_frames=3, min_px=1.5, max_px=20.0):
    """
    A fast asteroid crosses a 5x5 stamp in a frame or two, so each of its detections looks stationary -- but
    it leaves a short chain of OTHER objects' detections stepping across the image in the frames either side.
    Per event, the other-object detections (same cut and frame bin) within +/- max_frames frames and
    min_px-max_px away:
      tab_chain_n            how many there are
      tab_chain_cos          cosine of the angle between the best before / after pair as seen from this event
                             (-1: on opposite sides along a line, i.e. a track through it)
      tab_chain_speed_ratio  their speeds (px per frame), smaller over larger: ~1 for a steady track
    """
    from scipy.spatial import cKDTree

    cols = ['tab_chain_n', 'tab_chain_cos', 'tab_chain_speed_ratio']
    out = pd.DataFrame(np.nan, index=events.index, columns=cols)
    if not {'frame_max', 'frame_bin', 'xcentroid', 'ycentroid', 'objid'} <= set(events):
        return out
    keys = [k for k in ['sector', 'camera', 'ccd', 'cut', 'frame_bin'] if k in events]
    scale = max_px / max_frames                 # one frame counts as this many pixels in the search
    for idx in events.groupby(keys, sort=False).groups.values():
        g = events.loc[idx]
        x, y = g['xcentroid'].to_numpy(float), g['ycentroid'].to_numpy(float)
        f = g['frame_max'].to_numpy(float)
        obj = g['objid'].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
        pts = np.c_[x, y, f * scale]
        tree = cKDTree(pts[ok])
        okidx = np.flatnonzero(ok)
        res = np.full((len(g), 3), np.nan)
        for j, nb in zip(okidx, tree.query_ball_point(pts[ok], np.hypot(max_px, max_px))):
            nb = okidx[nb]
            df = f[nb] - f[j]
            dx, dy = x[nb] - x[j], y[nb] - y[j]
            d = np.hypot(dx, dy)
            keep = (obj[nb] != obj[j]) & (np.abs(df) >= 1) & (np.abs(df) <= max_frames) & (d >= min_px) & (d <= max_px)
            res[j, 0] = keep.sum()
            before, after = np.flatnonzero(keep & (df < 0)), np.flatnonzero(keep & (df > 0))
            if len(before) and len(after):
                vb = np.c_[dx[keep & (df < 0)], dy[keep & (df < 0)]]
                va = np.c_[dx[keep & (df > 0)], dy[keep & (df > 0)]]
                cos = (vb @ va.T) / np.outer(np.hypot(*vb.T), np.hypot(*va.T))
                ib, ia = np.unravel_index(np.argmin(cos), cos.shape)
                res[j, 1] = cos[ib, ia]
                sb = np.hypot(*vb[ib]) / abs(df[before[ib]])
                sa = np.hypot(*va[ia]) / abs(df[after[ia]])
                res[j, 2] = min(sb, sa) / max(sb, sa)
        out.loc[idx] = res
    return out


def _object_gap_cv(events):
    """
    Coefficient of variation of the gaps between an object's event peak times
    (same frame bin). Low values mean suspiciously regular recurrence.
    """
    keys = [k for k in ['sector', 'camera', 'ccd', 'cut', 'objid', 'frame_bin'] if k in events]
    if 'mjd_max' not in events or 'objid' not in keys:
        return pd.Series(np.nan, index=events.index)

    def _cv(t):
        t = np.sort(t.to_numpy(float))
        gaps = np.diff(t[np.isfinite(t)])
        if len(gaps) < 2 or np.mean(gaps) <= 0:
            return np.nan
        return np.std(gaps) / np.mean(gaps)

    return events.groupby(keys)['mjd_max'].transform(_cv)


def table_features(events):
    """
    Features from the event table itself (tab_*). Pass the full table for a cut
    so the simultaneity and per-object statistics see every event.
    """
    f = pd.DataFrame(index=events.index)

    def col(name):
        if name in events:
            return pd.to_numeric(events[name], errors='coerce')
        return pd.Series(np.nan, index=events.index)

    for name in ['frame_duration', 'n_detections', 'image_sig_max', 'lc_sig_max', 'lc_sig_med', 'lc_flat',
                 'snr_psf', 'psf_like', 'psf_diff', 'psf_stacked', 'ellipticity', 'fwhm', 'neg_extent',
                 'flux_sign', 'frame_bin', 'total_events', 'bad_frame_flag', 'known_asteroid_dist_px']:
        f[f'tab_{name}'] = col(name)

    f['tab_mjd_duration_hr'] = col('mjd_duration') * 24
    f['tab_det_frac'] = col('n_detections') / col('frame_duration')
    f['tab_log_flux_max'] = np.log10(np.abs(col('flux_max')) + 1e-3)
    f['tab_abs_gal_b'] = np.abs(col('gal_b'))
    f['tab_det_psf_offset'] = np.hypot(col('xcentroid_det') - col('xcentroid_psf'),
                                       col('ycentroid_det') - col('ycentroid_psf'))

    x, y = col('xccd'), col('yccd')     # science pixels: columns 44-2091, rows 0-2047
    f['tab_ccd_edge_dist'] = np.minimum(np.minimum(x - 44, 2091 - x), np.minimum(y, 2047 - y))

    mask = col('source_mask')
    finite = np.isfinite(mask)
    bits = np.where(finite, mask, 0).astype(int)
    for b in range(4):
        f[f'tab_source_mask_b{b}'] = np.where(finite, (bits >> b) & 1, np.nan)

    f['tab_n_crossbin'] = events['crossbin_ids'].apply(_n_crossbin) if 'crossbin_ids' in events else np.nan
    f['tab_obj_gap_cv'] = _object_gap_cv(events)
    return pd.concat([f, _time_crowding(events), _chain_features(events)], axis=1)


# ----------------------------- Light-curve features ----------------------------- #

def _robust_std(x):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _running_median(y, mask, window, bounds):
    """Centred running median per continuous segment, ignoring masked points."""
    s = pd.Series(np.where(mask, np.nan, y))
    trend = np.full(len(y), np.nan)
    for a, b in bounds:
        seg = s.iloc[a:b].rolling(window, center=True, min_periods=max(3, window // 5)).median()
        trend[a:b] = seg.interpolate(limit_direction='both').to_numpy()
    if np.isnan(trend).any():       # a segment with nothing unmasked: fall back to the light curve's median
        trend[np.isnan(trend)] = np.nanmedian(s.to_numpy())
    return trend


def _block_mean(y, k):
    """Mean of consecutive blocks of k values (NaNs ignored; a block needs at least half its values)."""
    if k <= 1:
        return np.asarray(y, float)
    m = len(y) // k
    if m == 0:
        return np.array([])
    blocks = np.asarray(y[:m * k], float).reshape(m, k)
    ok = np.isfinite(blocks).sum(axis=1) >= max(1, k // 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.where(ok, np.nanmean(blocks, axis=1), np.nan)


def _gauss(t, A, t0, sigma, c):
    return A * np.exp(-0.5 * ((t - t0) / sigma) ** 2) + c


def _dexp(t, A, t0, tau_rise, tau_fall, c):
    """Two-sided exponential: symmetric for asteroids/pulsations, fast-rise slow-decay for flares."""
    dt = t - t0
    rise = np.exp(np.clip(dt / tau_rise, -50, 0))
    fall = np.exp(np.clip(-dt / tau_fall, -50, 0))
    return A * np.where(dt < 0, rise, fall) + c


def _fit_features(idx, zw, ip, rise, decay, cadence):
    from scipy.optimize import curve_fit

    out = dict(lc_fit_gauss_r2=np.nan, lc_fit_dexp_r2=np.nan, lc_fit_dbic=np.nan,
               lc_fit_log_tau_ratio=np.nan, lc_fit_log_rise_days=np.nan, lc_fit_log_fall_days=np.nan)
    m = np.isfinite(zw)
    if m.sum() < 6:
        return out

    x = idx[m].astype(float)
    v = zw[m]
    k = len(v)
    span = x.max() - x.min() + 1
    amp = max(np.nanmax(v), 1e-3)
    sst = np.sum((v - v.mean()) ** 2)
    if sst <= 0:
        return out

    # scipy's default tolerances (1e-8) left ~2% of real-data fits running to maxfev and failing, which took most
    # of the fitting time; at 1e-5 they converge and the fitted values move by ~1e-4
    tol = dict(ftol=1e-5, xtol=1e-5, gtol=1e-5)
    rss = {}
    try:
        p, _ = curve_fit(_gauss, x, v, p0=[amp, ip, max(1.0, (rise + decay + 1) / 2.355), 0.0],
                         bounds=([0, x.min(), 0.3, -np.inf], [np.inf, x.max(), 3 * span, np.inf]), maxfev=2000, **tol)
        rss['gauss'] = np.sum((v - _gauss(x, *p)) ** 2)
    except Exception:
        pass
    try:
        p, _ = curve_fit(_dexp, x, v, p0=[amp, ip, max(rise, 0.5), max(decay, 0.5), 0.0],
                         bounds=([0, x.min(), 0.1, 0.1, -np.inf], [np.inf, x.max(), 3 * span, 3 * span, np.inf]),
                         maxfev=2000, **tol)
        rss['dexp'] = np.sum((v - _dexp(x, *p)) ** 2)
        out['lc_fit_log_tau_ratio'] = np.log10(p[3] / p[2])
        out['lc_fit_log_rise_days'] = np.log10(p[2] * cadence)
        out['lc_fit_log_fall_days'] = np.log10(p[3] * cadence)
    except Exception:
        pass

    if 'gauss' in rss:
        out['lc_fit_gauss_r2'] = 1 - rss['gauss'] / sst
    if 'dexp' in rss:
        out['lc_fit_dexp_r2'] = 1 - rss['dexp'] / sst
    if len(rss) == 2:
        bic = {name: k * np.log(max(r, 1e-12) / k) for name, r in rss.items()}
        out['lc_fit_dbic'] = (bic['dexp'] + 5 * np.log(k)) - (bic['gauss'] + 4 * np.log(k))
    return out


def _self_similarity(z, a, b, peak):
    """
    Matched filter of this event's shape against the rest of its own light
    curve: does a comparable-amplitude copy of the event occur elsewhere?
    """
    from numpy.lib.stride_tricks import sliding_window_view

    out = {'ctx_selfsim_max': np.nan, 'ctx_n_similar': np.nan}
    tpl = z[a:b + 1]
    if len(tpl) < 3 or np.isfinite(tpl).mean() < 0.8:
        return out
    tpl = np.nan_to_num(tpl)
    tpl = tpl - tpl.mean()
    norm = np.sqrt(np.sum(tpl ** 2))
    zz = np.nan_to_num(z)
    L, N = len(tpl), len(zz)
    if norm == 0 or N < 3 * L:
        return out

    num = np.correlate(zz, tpl, mode='valid')
    c1 = np.concatenate([[0.0], np.cumsum(zz)])
    c2 = np.concatenate([[0.0], np.cumsum(zz ** 2)])
    s1, s2 = c1[L:] - c1[:-L], c2[L:] - c2[:-L]
    ncc = num / (np.sqrt(np.maximum(s2 - s1 ** 2 / L, 1e-12)) * norm)

    lags = np.arange(N - L + 1)
    window_max = sliding_window_view(zz, L).max(axis=1)
    ok = (np.abs(lags - a) >= L) & (window_max >= 0.5 * peak)
    if not ok.any():
        out['ctx_selfsim_max'] = 0.0
        out['ctx_n_similar'] = 0
        return out

    cand = lags[ok]
    out['ctx_selfsim_max'] = float(np.max(ncc[cand]))
    blocked = np.zeros(len(lags), bool)          # lags within L of a copy already counted
    n_similar = 0
    for lag in cand[np.argsort(-ncc[cand])][:5000]:
        if ncc[lag] < 0.7:
            break
        if not blocked[lag]:
            n_similar += 1
            blocked[max(lag - L + 1, 0):lag + L] = True
    out['ctx_n_similar'] = n_similar
    return out


def _periodicity(t, z, use, t_peak, peak, duration_days, cfg):
    """Lomb-Scargle on the light curve outside this event, restricted to fast periods."""
    from astropy.timeseries import LombScargle

    out = dict(ctx_ls_power=np.nan, ctx_ls_log_fap=np.nan, ctx_ls_log_period=np.nan, ctx_ls_amp_z=np.nan,
               ctx_ls_amp_over_peak=np.nan, ctx_ls_model_at_peak=np.nan, ctx_ls_period_over_dur=np.nan)
    m = use & np.isfinite(z)
    if m.sum() < 50:
        return out
    tt, zz = t[m], z[m]
    cadence = np.nanmedian(np.diff(tt))
    pmin = max(cfg['ls_min_period_days'], 3 * cadence)
    pmax = min(cfg['ls_max_period_days'], (tt.max() - tt.min()) / 2)
    if pmax <= pmin:
        return out

    ls = LombScargle(tt, zz)
    freq, power = ls.autopower(minimum_frequency=1 / pmax, maximum_frequency=1 / pmin, samples_per_peak=5)
    i = int(np.argmax(power))
    f0, p0 = freq[i], power[i]
    fap = ls.false_alarm_probability(p0, minimum_frequency=1 / pmax, maximum_frequency=1 / pmin, method='baluev')
    cycle = ls.model(tt.min() + np.linspace(0, 1 / f0, 64), f0)
    amp = (cycle.max() - cycle.min()) / 2

    out['ctx_ls_power'] = p0
    out['ctx_ls_log_fap'] = np.log10(max(fap, 1e-300))
    out['ctx_ls_log_period'] = np.log10(1 / f0)
    out['ctx_ls_amp_z'] = amp
    out['ctx_ls_amp_over_peak'] = amp / peak
    out['ctx_ls_model_at_peak'] = (ls.model(np.array([t_peak]), f0)[0] - cycle.mean()) / peak
    out['ctx_ls_period_over_dur'] = (1 / f0) / max(duration_days, cadence)
    return out


def _lc_features(t, lc, fs, fe, cadence, bounds, cfg):
    from scipy.signal import find_peaks

    out = {}
    n = len(lc)
    dur = fe - fs + 1
    # hide the event plus a margin before estimating the baseline -- capped, or a multi-day event's
    # margin swallows a whole orbit and leaves no baseline at all
    pad = max(3, min(dur, int(round(cfg['max_mask_pad_days'] / cadence))))
    a, b = max(fs - pad, 0), min(fe + pad, n - 1)
    in_window = np.zeros(n, bool)
    in_window[a:b + 1] = True

    # -- Detrend the sector light curve without letting this event pull the trend -- #
    window = max(5, int(round(cfg['detrend_days'] / cadence)) | 1)
    det = lc - _running_median(lc, in_window, window, bounds)
    outside = ~in_window & np.isfinite(det)
    if outside.sum() < 20:
        return out

    # Noise from point-to-point scatter: unlike the MAD of the light curve it is not inflated
    # by stellar variability, so a variable's other cycles register at their real significance.
    sig = _robust_std(np.diff(np.where(outside, det, np.nan))) / np.sqrt(2)
    sig_mad = _robust_std(det[outside])
    if not (np.isfinite(sig) and sig > 0 and np.isfinite(sig_mad) and sig_mad > 0):
        return out
    z = det / sig

    # -- Character of the pixel away from the event -- #
    zo = z[outside]
    local = outside & (np.abs(t - t[fs]) < 0.5)
    sig_local = _robust_std(det[local]) if local.sum() >= 10 else np.nan
    out['ctx_log_sig'] = np.log10(sig)
    out['ctx_var_ratio'] = sig_mad / sig                      # >1: the pixel varies beyond its noise
    out['ctx_sig_local_ratio'] = sig_local / sig_mad
    out['ctx_std_over_mad'] = np.std(det[outside]) / sig_mad  # ~1 for Gaussian scatter, larger for heavy tails
    out['ctx_frac_pos5'] = np.mean(zo > 5)
    out['ctx_frac_neg5'] = np.mean(zo < -5)

    edges = np.array([t[s] for bound in bounds for s in (bound[0], bound[1] - 1)])
    out['ctx_edge_dist_days'] = float(np.min(np.minimum(np.abs(edges - t[fs]), np.abs(edges - t[fe]))))

    # -- Does the event window run across missing data (a downlink or a gap)? Almost always junk -- #
    dt_ev = np.diff(t[fs:fe + 1])
    gap = dt_ev > 1.5 * cadence
    out['lc_gap_frames'] = float(np.sum(np.round(dt_ev[gap] / cadence) - 1)) if gap.any() else 0.0
    out['lc_spans_gap'] = float(any(fs < s <= fe for bound in bounds for s in (bound[0],)))
    out['lc_event_nan_frac'] = float(np.mean(~np.isfinite(lc[fs:fe + 1])))

    # -- Variability on ~hour timescales after the day-scale detrend (stochastic variables), as a
    #    multiple of what white noise at this level would give -- #
    k = max(1, int(round((1 / 24) / cadence)))
    binned = _block_mean(np.where(outside, det, np.nan), k)
    if np.isfinite(binned).sum() >= 10:
        out['ctx_red_noise'] = _robust_std(binned[np.isfinite(binned)]) / (sig / np.sqrt(k))

    # -- Shape of the event itself -- #
    ze = z[fs:fe + 1]
    if not np.isfinite(ze).any():
        return out
    ip = fs + int(np.nanargmax(ze))
    peak = z[ip]
    out['lc_peak_z'] = peak
    out['lc_peak_z_local'] = peak * sig / sig_local
    out['lc_mean_z'] = np.nanmean(ze)
    out['lc_frac_above3'] = np.mean(ze > 3)
    out['lc_valid_frac'] = np.isfinite(z[a:b + 1]).mean()
    out['lc_peak_position'] = (ip - fs) / max(dur - 1, 1)
    if not peak > 0:
        return out

    half = peak / 2
    lo, hi = ip, ip
    while lo - 1 >= a and z[lo - 1] >= half:
        lo -= 1
    while hi + 1 <= b and z[hi + 1] >= half:
        hi += 1
    rise, decay = ip - lo, hi - ip
    out['lc_rise_frames'] = rise
    out['lc_decay_frames'] = decay
    out['lc_asym'] = (decay - rise) / (decay + rise) if decay + rise > 0 else 0.0
    out['lc_fwhm_frac'] = (hi - lo + 1) / dur
    out['lc_log_fwhm_days'] = np.log10((hi - lo + 1) * cadence)
    out['lc_spikiness'] = peak / np.nansum(np.clip(ze, 0, None))

    # -- One-frame spikes: how far the peak stands above its neighbours, and how strong
    #    the event is without that frame (would it have been detected at all?) -- #
    sides = [z[j] for j in (ip - 1, ip + 1) if 0 <= j < n and np.isfinite(z[j])]
    out['lc_peak_over_neighbour'] = peak / max(max(sides), 1.0) if sides else np.nan
    rest = np.delete(ze, ip - fs)
    rest = rest[np.isfinite(rest)]
    out['lc_despiked_peak_z'] = float(np.max(rest)) if rest.size else 0.0
    out['lc_despiked_peak_ratio'] = out['lc_despiked_peak_z'] / peak

    zw = z[a:b + 1]
    pk, props = find_peaks(np.nan_to_num(zw), prominence=1.0)
    pk_abs = pk + a
    in_event = (pk_abs >= fs) & (pk_abs <= fe)
    out['lc_n_peaks'] = int(np.sum(in_event & (props['prominences'] >= 2)))
    others = zw[pk][pk_abs != ip]
    out['lc_second_peak_ratio'] = float(np.max(others)) / peak if others.size else 0.0

    pre, post = z[a:fs], z[fe + 1:b + 1]
    out['lc_pre_z'] = np.nanmedian(pre) if np.isfinite(pre).sum() >= 2 else np.nan
    out['lc_post_z'] = np.nanmedian(post) if np.isfinite(post).sum() >= 2 else np.nan
    out['lc_step_z'] = out['lc_post_z'] - out['lc_pre_z']
    out['lc_min_z'] = np.nanmin(zw)
    out['lc_neg_ratio'] = -np.nanmin(zw) / peak

    out.update(_fit_features(np.arange(a, b + 1), zw, ip, rise, decay, cadence))

    # -- Duration-normalised shape: same number of points for a 2-frame and a 40-frame event -- #
    grid = np.linspace(fs - dur, fe + dur, cfg['n_shape'])
    valid = np.isfinite(z)
    shape = np.full(len(grid), np.nan)
    inside = (grid >= 0) & (grid <= n - 1)
    if valid.sum() >= 2:
        shape[inside] = np.interp(grid[inside], np.flatnonzero(valid), z[valid]) / peak
    for i, v in enumerate(shape):
        out[f'shape_{i:02d}'] = v

    # -- Is the event unusual within its own sector light curve? -- #
    zc = np.where(outside, z, np.nan)
    span_days = max(outside.sum() * cadence, cadence)
    h = find_peaks(np.nan_to_num(zc), height=max(3.0, 0.3 * peak), distance=max(1, dur))[1]['peak_heights']
    neg = find_peaks(np.nan_to_num(-zc), height=max(3.0, 0.5 * peak), distance=max(1, dur))[1]['peak_heights']
    out['ctx_n_comparable50'] = int(np.sum(h >= 0.5 * peak))
    out['ctx_n_comparable80'] = int(np.sum(h >= 0.8 * peak))
    out['ctx_rate_comparable50'] = out['ctx_n_comparable50'] / span_days
    out['ctx_n_neg_comparable50'] = int(neg.size)
    out['ctx_peak_over_max_other'] = peak / np.nanmax(zc) if np.nanmax(zc) > 0 else np.nan
    # against the star's usual range rather than its single largest excursion (robust to one outlier)
    p99 = np.nanpercentile(zc, 99) if np.isfinite(zc).sum() >= 100 else np.nan
    out['ctx_peak_over_p99'] = peak / p99 if p99 > 0 else np.nan

    out.update(_self_similarity(z, a, b, peak))
    out.update(_periodicity(t, z, outside, t[ip], peak, dur * cadence, cfg))
    return out


# ----------------------------- Pixel features ----------------------------- #

_NEIGHBOURS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]


def _gauss_corr(img):
    """Correlation of a 5x5 image with a symmetric Gaussian at its flux-weighted centroid."""
    yy, xx = np.mgrid[0:5, 0:5]
    w = np.clip(np.nan_to_num(img), 0, None)
    fin = np.isfinite(img)
    if w.sum() <= 0 or fin.sum() < 9 or np.std(img[fin]) == 0:
        return np.nan
    cy, cx = np.sum(w * yy) / w.sum(), np.sum(w * xx) / w.sum()
    g = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 0.9 ** 2)
    return np.corrcoef(img[fin], g[fin])[0, 1]


def _centroid_track(ev, frames):
    """
    Flux-weighted centroid path of a 5x5 stamp sequence through the given frames
    that are at least half as bright as the brightest of them.
    """
    yy, xx = np.mgrid[0:5, 0:5]
    total = np.nansum(ev[:, 1:4, 1:4], axis=(1, 2))
    frames = [k for k in frames if np.isfinite(total[k])]
    if not frames or np.max(total[frames]) <= 0:
        return {}
    top = np.max(total[frames])
    cen = []
    for k in frames:
        w = np.clip(np.nan_to_num(ev[k]), 0, None)
        if total[k] >= 0.5 * top and w.sum() > 0:
            cen.append((np.sum(w * xx) / w.sum() - 2, np.sum(w * yy) / w.sum() - 2, k))
    out = {}
    if cen:
        c = np.array(cen)
        out['mean_offset'] = float(np.hypot(c[:, 0].mean(), c[:, 1].mean()))
    if len(cen) >= 2:
        steps = np.diff(c[:, :2], axis=0)
        path = float(np.sum(np.hypot(steps[:, 0], steps[:, 1])))
        disp = float(np.hypot(*(c[-1, :2] - c[0, :2])))
        out.update(disp=disp, path=path, straightness=disp / path if path > 0 else np.nan,
                   speed=disp / max(c[-1, 2] - c[0, 2], 1))
    return out


def _pixel_features(st, fs, fe, fm, noise):
    """Features of the 5x5 pixel stamp (already multiplied by flux_sign) through the event."""
    out = {}
    nt = st.shape[0]

    # -- Brightest frame -- #
    img = st[fm]
    core = img[1:4, 1:4]
    pos_core = np.nansum(np.clip(core, 0, None))
    if np.isfinite(img).sum() >= 9 and pos_core > 0:
        out['pix_peak_frac'] = np.nanmax(core) / pos_core
        iy, ix = np.unravel_index(np.nanargmax(img), img.shape)
        out['pix_peak_offset'] = float(np.hypot(iy - 2, ix - 2))
        pos = np.nansum(np.clip(img, 0, None))
        out['pix_neg_frac'] = -np.nansum(np.clip(img, None, 0)) / pos
        out['pix_outer_ratio'] = (np.nansum(img) - np.nansum(core)) / pos_core
        out['pix_gauss_corr'] = _gauss_corr(img)
        if noise is not None and noise[fm] > 0:
            out['pix_peak_pixel_z'] = np.nanmax(core) / noise[fm]

        # What the brightest frame adds over its neighbours: PSF-shaped for a real brightening,
        # sharp and concentrated for a cosmic ray -- alone, or landing on another event
        sides = [st[j] for j in (fm - 1, fm + 1) if 0 <= j < nt and np.isfinite(st[j]).any()]
        if sides:
            excess = img - np.nanmean(sides, axis=0)
            ecore = np.nansum(np.clip(excess[1:4, 1:4], 0, None))
            out['pix_excess_frac'] = ecore / pos_core
            if ecore > 0:
                out['pix_excess_peak_frac'] = np.nanmax(excess[1:4, 1:4]) / ecore
                out['pix_excess_gauss_corr'] = _gauss_corr(excess)

    # -- Do the PSF pixels brighten together (a real point source) or alone (a hit)? -- #
    lo, hi = max(fs - 1, 0), min(fe + 1, nt - 1)
    seq = st[lo:hi + 1]
    if len(seq) >= 4:
        centre = seq[:, 2, 2]
        cors = []
        for j, i in _NEIGHBOURS:
            nb = seq[:, j, i]
            m = np.isfinite(centre) & np.isfinite(nb)
            if m.sum() >= 4 and np.std(centre[m]) > 0 and np.std(nb[m]) > 0:
                cors.append(np.corrcoef(centre[m], nb[m])[0, 1])
        out['pix_coherence'] = float(np.median(cors)) if cors else np.nan

    ev = st[fs:fe + 1]
    evcore = ev[:, 1:4, 1:4]
    csum = np.nansum(np.clip(evcore, 0, None), axis=(1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.nanmax(evcore.reshape(len(ev), -1), axis=1) / np.where(csum > 0, csum, np.nan)
    if np.isfinite(frac).any():
        out['pix_single_frac_max'] = float(np.nanmax(frac))

    # -- Centroid motion through the bright part of the event (asteroids move, stars don't) -- #
    track = _centroid_track(ev, range(len(ev)))
    for k in ['mean_offset', 'disp', 'path', 'straightness', 'speed']:
        if k in track:
            out[f'pix_cen_{k}'] = track[k]
    # ... and without the brightest frame, so a cosmic ray on top doesn't hide the motion underneath
    track = _centroid_track(ev, [k for k in range(len(ev)) if k != fm - fs])
    out['pix_cen_disp_nopeak'] = track.get('disp', np.nan)
    out['pix_cen_straightness_nopeak'] = track.get('straightness', np.nan)
    return out


def _mean_image(st, lo, hi):
    """Mean of stamp frames [lo, hi) clipped to the series, or None if there are none."""
    lo, hi = max(lo, 0), min(hi, st.shape[0])
    if hi <= lo or not np.isfinite(st[lo:hi]).any():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(st[lo:hi], axis=0)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 9 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _centroid5(img):
    """Flux-weighted (positive pixels) centroid of a 5x5 image, relative to its centre."""
    yy, xx = np.mgrid[0:5, 0:5]
    w = np.clip(np.nan_to_num(img), 0, None)
    if w.sum() <= 0:
        return None
    return np.sum(w * xx) / w.sum() - 2, np.sum(w * yy) / w.sum() - 2


def _epoch_features(st, fs, fe, fm, noise):
    """
    The image at three epochs: just BEFORE the event, its PEAK, and just AFTER -- defined for every event,
    one frame long or forty. A real new source is absent before and after; a residual of a star that is
    always there (poor subtraction, a variable's other cycles) shows the same pattern before and after.
    For events of 3+ frames, the FIRST half against the SECOND half: an asteroid's image shifts, a star's
    doesn't. Stacking each half averages down the noise that makes frame-to-frame centroids jump around.
    st: 9x9 stamp series (already multiplied by flux_sign).
    """
    out = {}
    c = st.shape[1] // 2
    inner = slice(c - 2, c + 3)       # the 5x5 around the event
    core = slice(c - 1, c + 2)        # its 3x3 core
    dur = fe - fs + 1
    nb = int(np.clip(dur, 3, 10))

    peak = st[fm]
    pk_core = np.nansum(np.clip(peak[core, core], 0, None))
    if not pk_core > 0:
        return out
    pk5 = peak[inner, inner]
    for name, img, frames in [('before', _mean_image(st, fs - nb, fs), np.arange(fs - nb, fs)),
                              ('after', _mean_image(st, fe + 1, fe + 1 + nb), np.arange(fe + 1, fe + 1 + nb))]:
        if img is None:
            continue
        out[f'pix_ep_{name}_frac'] = float(np.nansum(img[core, core]) / pk_core)
        out[f'pix_ep_{name}_corr'] = _corr(img[inner, inner], pk5)
        if noise is not None:
            fr = frames[(frames >= 0) & (frames < len(noise))]
            nz = np.nanmedian(noise[fr]) / np.sqrt(len(fr)) if len(fr) else np.nan
            if nz > 0:
                out[f'pix_ep_{name}_absz'] = float(np.nanmean(np.abs(img[core, core])) / nz)

    if dur >= 3:
        half = -(-dur // 2)
        first = _mean_image(st, fs, fs + half)
        second = _mean_image(st, fe + 1 - half, fe + 1)
        if first is not None and second is not None:
            a, b = first[inner, inner], second[inner, inner]
            ca, cb = _centroid5(a), _centroid5(b)
            if ca is not None and cb is not None:
                out['pix_ep_shift'] = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
            zero = _corr(a, b)
            best, best_shift = zero, 0.0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    r = _corr(a, second[c - 2 + dy:c + 3 + dy, c - 2 + dx:c + 3 + dx])
                    if np.isfinite(r) and (not np.isfinite(best) or r > best):
                        best, best_shift = r, float(np.hypot(dx, dy))
            out['pix_ep_corr_zero'] = zero
            out['pix_ep_corr_gain'] = best - zero if np.isfinite(best) and np.isfinite(zero) else np.nan
            out['pix_ep_best_shift'] = best_shift
            g = [_gauss_corr(a), _gauss_corr(pk5), _gauss_corr(b)]
            out['pix_ep_gauss_first'], out['pix_ep_gauss_last'] = g[0], g[2]
            out['pix_ep_gauss_min'] = float(np.nanmin(g)) if np.isfinite(g).any() else np.nan
    return out


def _ring_lc_features(t, st, fs, fe, cadence, bounds, cfg):
    """
    The event pixel's light curve against the ring of pixels around it, away from the event, on ~1 h bins
    after the day-scale detrend: scattered light and poor subtraction move the whole neighbourhood together
    (high correlation, ratio ~1); a variable star varies at its own position.
    """
    out = {}
    n, size = st.shape[0], st.shape[1]
    c = size // 2
    ring_mask = np.ones((size, size), bool)
    ring_mask[c - 2:c + 3, c - 2:c + 3] = False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        core = np.nanmean(st[:, c - 1:c + 2, c - 1:c + 2], axis=(1, 2))
        ring = np.nanmean(st[:, ring_mask], axis=1)
    dur = fe - fs + 1
    # hide the event plus a margin before estimating the baseline -- capped, or a multi-day event's
    # margin swallows a whole orbit and leaves no baseline at all
    pad = max(3, min(dur, int(round(cfg['max_mask_pad_days'] / cadence))))
    in_window = np.zeros(n, bool)
    in_window[max(fs - pad, 0):min(fe + pad, n - 1) + 1] = True
    window = max(5, int(round(cfg['detrend_days'] / cadence)) | 1)
    k = max(1, int(round((1 / 24) / cadence)))
    det = []
    for lc in (core, ring):
        d = lc - _running_median(lc, in_window, window, bounds)
        det.append(_block_mean(np.where(in_window, np.nan, d), k))
    bc, br = det
    m = np.isfinite(bc) & np.isfinite(br)
    if m.sum() >= 10:
        out['ctx_ring_corr'] = _corr(bc[m], br[m]) if m.sum() >= 9 else np.nan
        sr = _robust_std(br[m])
        out['ctx_core_ring_ratio'] = _robust_std(bc[m]) / sr if sr > 0 else np.nan
    return out


def _ring_features(img, noise_fm):
    """Extended structure around the event at its brightest frame (glints, scattered light)."""
    out = {}
    c = img.shape[0] // 2
    inner = np.zeros(img.shape, bool)
    inner[c - 2:c + 3, c - 2:c + 3] = True
    ring = img[~inner & np.isfinite(img)]
    if ring.size < 20:
        return out
    scale = noise_fm if (noise_fm is not None and noise_fm > 0) else _robust_std(ring)
    if np.isfinite(scale) and scale > 0:
        out['pix_ring_frac3'] = float(np.mean(ring > 3 * scale))
        out['pix_ring_median_z'] = float(np.median(ring) / scale)
    return out


def _event_features(ev, cd, cfg):
    out = {}
    fb = max(int(ev['frame_bin']), 1) if np.isfinite(ev.get('frame_bin', 1)) else 1
    sign = -1.0 if ev.get('flux_sign', 1) < 0 else 1.0
    t, noise, bounds = cd.binned(fb)
    nt = len(t)
    fs = int(np.clip(ev['frame_start'], 0, nt - 1))
    fe = int(np.clip(ev['frame_end'], fs, nt - 1))
    fm = int(np.clip(ev['frame_max'], fs, fe))
    x, y = int(ev['xint']), int(ev['yint'])
    cadence = np.nanmedian(np.diff(t))

    st = cd.stamp(x, y, 2, fb) * sign
    core = st[:, 1:4, 1:4]
    lc = np.nansum(core, axis=(1, 2))
    lc[~np.isfinite(core).any(axis=(1, 2))] = np.nan

    out.update(_lc_features(t, lc, fs, fe, cadence, bounds, cfg))
    out.update(_pixel_features(st, fs, fe, fm, noise))
    st9 = cd.stamp(x, y, 4, fb) * sign
    out.update(_epoch_features(st9, fs, fe, fm, noise))
    out.update(_ring_lc_features(t, st9, fs, fe, cadence, bounds, cfg))
    out.update(_ring_features(cd.image(x, y, 7, fb, fm) * sign, None if noise is None else noise[fm]))
    if noise is not None:
        med = np.nanmedian(noise)
        out['ctx_frame_noise_ratio'] = noise[fm] / med
        out['ctx_frame_noise_ratio_max'] = np.nanmax(noise[fs:fe + 1]) / med
        # an unstable stretch of the cut: mean frame noise over the hours around the event
        for hours in (3, 12):
            out[f'ctx_frame_noise_{hours}h'] = np.nanmean(noise[np.abs(t - t[fm]) <= hours / 24]) / med
    return out


# ----------------------------- Catalogue features ----------------------------- #

_XM_COLS = ['xm_gaia_sep', 'xm_gaia_sep_norm', 'xm_gaia_mag', 'xm_gaia_n2px', 'xm_gaia_brightest_1px', 'xm_var_sep']


def _crossmatch_features(events, data_path, sector, cam, ccd, cut, n=8):
    """
    Distance to the nearest Gaia / variable-catalogue source for every event,
    from the same local catalogues and CutWCS as Detector._catalogue_crossmatch
    (which skips events already tagged Asteroid/CosmicRay/Junk).
    """
    from scipy.spatial import cKDTree

    out = pd.DataFrame(np.nan, index=events.index, columns=_XM_COLS)
    path = _cut_path(data_path, sector, cam, ccd, cut, n)
    try:
        from .localisation import CutWCS
        wcs = CutWCS(data_path, sector, cam, ccd, cut, n)
        gaia = pd.read_csv(f'{path}/local_gaia_cat.csv')
    except Exception as e:
        _warn_once(('xm', data_path), f'Catalogue features unavailable ({type(e).__name__}: {e}); xm_* left NaN.')
        return out

    x = pd.to_numeric(events['xcentroid'], errors='coerce').to_numpy()
    y = pd.to_numeric(events['ycentroid'], errors='coerce').to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any() or len(gaia) == 0:
        return out
    pts = np.c_[x[ok], y[ok]]

    gx, gy = wcs.all_world2pix(gaia['ra'].to_numpy(), gaia['dec'].to_numpy(), 0)
    mag = gaia['Gmag'].to_numpy(float) if 'Gmag' in gaia else np.full(len(gaia), np.nan)
    if 'RPmag' in gaia:
        rp = gaia['RPmag'].to_numpy(float)
        mag = np.where(np.isfinite(rp), rp, mag)
    tree = cKDTree(np.c_[gx, gy])
    d, i = tree.query(pts, k=1)
    if 'centroid_err' in events:
        # radial-localisation outputs: one 1-sigma error per axis (older outputs: per-axis 95% radii, so
        # xm_gaia_sep_norm isn't on the same scale across the two; xm_ features are left out of the model)
        err = np.sqrt(2) * pd.to_numeric(events['centroid_err'], errors='coerce').to_numpy()[ok]
    else:
        err = np.hypot(pd.to_numeric(events['xcentroid_err'], errors='coerce'),
                       pd.to_numeric(events['ycentroid_err'], errors='coerce')).to_numpy()[ok]
    idx = events.index[ok]
    out.loc[idx, 'xm_gaia_sep'] = d
    out.loc[idx, 'xm_gaia_sep_norm'] = d / err
    out.loc[idx, 'xm_gaia_mag'] = mag[i]
    near = tree.query_ball_point(pts, 2.0)
    out.loc[idx, 'xm_gaia_n2px'] = [len(v) for v in near]
    close = tree.query_ball_point(pts, 1.0)
    out.loc[idx, 'xm_gaia_brightest_1px'] = [np.nanmin(mag[v]) if len(v) else np.nan for v in close]

    try:
        from .tools import load_table
        variables = load_table(f'{path}/variable_catalog.csv')
        if len(variables):
            vx, vy = wcs.all_world2pix(variables['ra'].to_numpy(), variables['dec'].to_numpy(), 0)
            out.loc[idx, 'xm_var_sep'] = cKDTree(np.c_[vx, vy]).query(pts, k=1)[0]
    except Exception:
        pass
    return out


# ----------------------------- Feature extraction ----------------------------- #

def _feature_columns(columns, groups=FEATURE_GROUPS, exclude=()):
    prefixes = tuple(f'{g}_' for g in groups)
    return [c for c in columns if c.startswith(prefixes) and not c.startswith(tuple(exclude))]


def _select_events(all_events, events, cfg, sector, cam, ccd, cut):
    """The events of a cut that get features: those in `events` (all if None), keeping at most cfg['max_tagged']
    of each pipeline tag (a fixed random choice per cut)."""
    selected = all_events
    if events is not None:
        wanted = events[['objid', 'eventid']].drop_duplicates()
        keep = all_events.reset_index().merge(wanted, on=['objid', 'eventid'])['index']
        selected = all_events.loc[keep]
    if cfg['max_tagged'] is not None:
        rng = np.random.default_rng([sector, cam, ccd, cut])
        drop = []
        for cls in PIPELINE_CLASSES:
            idx = selected.index[selected['classification'] == cls]
            if len(idx) > cfg['max_tagged']:
                drop += list(rng.choice(idx, len(idx) - cfg['max_tagged'], replace=False))
        selected = selected.drop(drop)
    return selected


def _event_features_safe(ev, cd, cfg):
    """_event_features for one event, or None if it fails. Kept at module level so joblib workers can run it;
    warnings are silenced in here because a worker process doesn't inherit the caller's warning filters."""
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            return _event_features(ev, cd, cfg)
        except Exception:
            return None


def _features_chunk(evs, cd, cfg):
    """Features for a list of events -- one joblib task, so the cut data is sent to a worker once per chunk."""
    return [_event_features_safe(ev, cd, cfg) for ev in evs]


def extract_cut_features(data_path, sector, cam, ccd, cut, n=8, events=None, config=None, all_events=None,
                         progress=False, n_jobs=1):
    """
    Feature table for the events of one cut.

    events : optional table (any columns incl. objid/eventid) restricting which
        events to compute; statistics that need the whole cut (simultaneity,
        per-object recurrence) still use every event in detected_events.csv.
    all_events : the cut's whole event table, if already in memory (e.g. the detector's, before it is
        saved); None = read detected_events.csv.
    progress : show a tqdm progress bar over the events (off by default: parallel callers would print one each).
    n_jobs : joblib workers over the events (1 = a plain loop), at most ML_MAX_JOBS. Each worker gets
        CHUNKS_PER_JOB chunks of events. Leave at 1 when the caller is already running cuts in parallel
        (extract_sector_features), or the workers multiply.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if all_events is None:
        all_events = load_cut_events(data_path, sector, cam, ccd, cut, n)
    else:
        all_events = _as_loaded(all_events)
    tab = table_features(all_events)
    selected = _select_events(all_events, events, cfg, sector, cam, ccd, cut)

    cd = _CutData(data_path, sector, cam, ccd, cut, n, frame_stats=cfg['frame_stats'])
    evs = [ev for _, ev in selected.iterrows()]
    n_jobs = max(1, min(n_jobs, ML_MAX_JOBS))
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=len(evs), desc='ML features')
    if n_jobs == 1:
        rows = []
        for ev in evs:
            rows.append(_event_features_safe(ev, cd, cfg))
            if progress:
                bar.update()
    else:
        # -- A few chunks of events per worker; the flux cube is memory-mapped, so workers re-open the file -- #
        from joblib import Parallel, delayed
        chunks = [c for c in np.array_split(np.arange(len(evs)), n_jobs * CHUNKS_PER_JOB) if len(c)]
        tasks = (delayed(_features_chunk)([evs[i] for i in c], cd, cfg) for c in chunks)
        try:
            parallel = Parallel(n_jobs=n_jobs, return_as='generator')   # results as chunks finish, in order
        except TypeError:                                                # joblib < 1.3: all at the end
            parallel = Parallel(n_jobs=n_jobs)
        rows = []
        for part in parallel(tasks):
            rows += part
            if progress:
                bar.update(len(part))
    if progress:
        bar.close()
    failed = sum(r is None for r in rows)
    rows = [{} if r is None else r for r in rows]
    if failed:
        print(f'  S{sector} C{cam} C{ccd} cut {cut}: feature extraction failed for {failed}/{len(selected)} events')

    parts = [selected[[c for c in META_COLS if c in selected]], tab.loc[selected.index],
             pd.DataFrame(rows, index=selected.index)]
    if cfg['crossmatch']:
        from astropy.wcs import FITSFixedWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)   # astropy tidying the WCS header's dates
            parts.append(_crossmatch_features(selected, data_path, sector, cam, ccd, cut, n))
    out = pd.concat(parts, axis=1)

    meta = [c for c in META_COLS if c in out]
    feats = sorted(_feature_columns(out.columns), key=lambda c: (FEATURE_GROUPS.index(c.split('_')[0]), c))
    return out[meta + feats].reset_index(drop=True)


def _count_rows(path):
    """Complete data rows in a csv (a row cut off mid-write doesn't count)."""
    with open(path, 'rb') as f:
        return max(sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1 << 20), b'')) - 1, 0)


def _cut_job(data_path, sector, cam, ccd, cut, n, events, config, cache, return_table=True):
    from .tools import table_exists
    if not table_exists(f'{_cut_path(data_path, sector, cam, ccd, cut, n)}/detected_events.csv'):
        return None
    if events is not None:
        events = events[(events.camera == cam) & (events.ccd == ccd) & (events.cut == cut)]
        if len(events) == 0:
            return None

    def summary(n_events):
        return pd.DataFrame([{'sector': sector, 'camera': cam, 'ccd': ccd, 'cut': cut, 'n_events': n_events}])

    if cache is not None and os.path.exists(cache):
        # a job killed while writing leaves a short file: only trust one with every expected event
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        expected = len(_select_events(load_cut_events(data_path, sector, cam, ccd, cut, n), events, cfg,
                                      sector, cam, ccd, cut))
        n_rows = _count_rows(cache)
        if n_rows == expected:
            return pd.read_csv(cache, low_memory=False) if return_table else summary(n_rows)
        print(f'  S{sector} C{cam} C{ccd} cut {cut}: cached file has {n_rows} of {expected} events; recomputing')
    try:
        df = extract_cut_features(data_path, sector, cam, ccd, cut, n, events=events, config=config)
    except Exception as e:
        print(f'  S{sector} C{cam} C{ccd} cut {cut}: skipped ({type(e).__name__}: {e})')
        return None
    if cache is not None:
        df.to_csv(f'{cache}.part', index=False)
        os.replace(f'{cache}.part', cache)          # the cache file only appears once it's complete
    return df if return_table else summary(len(df))


def build_feature_table(data_path='/fred/oz335/TESSdata', sector=None, cams=(1, 2, 3, 4), ccds=(1, 2, 3, 4),
                        cuts=None, n=8, events=None, cache_dir=None, overwrite=False, n_jobs=1, config=None,
                        return_table=True):
    """
    Feature table for every event across cuts. Each cut's flux cube is read
    once (memory-mapped); with cache_dir, per-cut results are saved so an
    interrupted run resumes where it stopped.

    events : optional table (sector/camera/ccd/cut/objid/eventid) restricting
        the computation, e.g. to the manually labelled events.
    return_table : False = only fill cache_dir, and return the number of events
        per cut. A whole sector is tens of millions of events -- too many to
        hold in memory; development/ml_collect_features.py picks the rows
        needed for training from the cache.
    """
    if not return_table and cache_dir is None:
        raise ValueError('return_table=False needs a cache_dir to write to.')
    from joblib import Parallel, delayed
    from tqdm import tqdm

    cuts = range(1, n ** 2 + 1) if cuts is None else cuts
    if events is not None:
        events = events[events.sector == sector]
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    jobs = []
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                cache = None
                if cache_dir is not None:
                    cache = f'{cache_dir}/S{sector}C{cam}C{ccd}C{cut}_features_v{FEATURE_VERSION}.csv'
                    if overwrite and os.path.exists(cache):
                        os.remove(cache)
                jobs.append((data_path, sector, cam, ccd, cut, n, events, config, cache, return_table))

    if n_jobs == 1:
        results = [_cut_job(*job) for job in tqdm(jobs, desc=f'Sector {sector} features')]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_cut_job)(*job) for job in tqdm(jobs, desc=f'Sector {sector} features'))

    results = [r for r in results if r is not None and len(r)]
    if not results:
        raise ValueError('No features extracted -- check data_path / sector / cuts.')
    return pd.concat(results, ignore_index=True)


# ----------------------------- Labels ----------------------------- #

def load_manual_labels(sort_dir, rename=None):
    """
    Labels from a manual sort (development/manual_sort.py or tools.manual_sort):
    one folder per group holding the sorted PNGs
    (S{s}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png) and an events.csv of their
    rows. The PNG names are the record of what was sorted (the csv can miss
    rows); the csv supplies position/time for re-matching. Events sorted into
    more than one group are dropped.

    rename : dict mapping folder names to class names, e.g.
        {'Other': 'Interesting', 'Cosmic Ray': 'CosmicRay'}; map a folder to
        None to leave it out (e.g. {'Unsure': None}). Names outside CLASSES
        become classes of their own and count as astrophysical.
    """
    rename = rename or {}
    rows, info = [], []
    for group in sorted(os.listdir(sort_dir)):
        gdir = os.path.join(sort_dir, group)
        if not os.path.isdir(gdir):
            continue
        label = rename.get(group, group)
        if label is None:
            continue
        for name in os.listdir(gdir):
            m = _IMAGE_NAME.match(name)
            if m:
                rows.append({**dict(zip(KEY_COLS, map(int, m.groups()))), 'label': label})
        csv = os.path.join(gdir, 'events.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            if {'sector', 'camera', 'ccd', 'cut', 'objid', 'eventid'} <= set(df):
                info.append(df[[c for c in KEY_COLS + ['frame_bin', 'xcentroid', 'ycentroid', 'mjd_max'] if c in df]])

    if not rows:
        raise ValueError(f'No sorted images found under {sort_dir}')
    labels = pd.DataFrame(rows).drop_duplicates()
    unknown = sorted(set(labels.label) - set(CLASSES))
    if unknown:
        warnings.warn(f'Groups not in {CLASSES} are kept as their own classes and counted as astrophysical: '
                      f'{unknown}. Use rename= to map them.')

    conflict = labels.duplicated(KEY_COLS, keep=False)
    if conflict.any():
        warnings.warn(f'{conflict.sum()} rows belong to events sorted into several groups; dropping them.')
        labels = labels[~conflict]

    if info:
        info = pd.concat(info, ignore_index=True).drop_duplicates(KEY_COLS)
        labels = labels.merge(info, on=KEY_COLS, how='left')
    return labels.reset_index(drop=True)


def _parse_ids(ids):
    import ast
    if isinstance(ids, str):
        ids = ast.literal_eval(ids) if ids.strip() else []
    return list(ids) if isinstance(ids, (list, tuple, np.ndarray)) else []


def attach_labels(features, manual=None, pipeline_weight=0.3, crossbin_weight=0.5, rematch=True, tol_px=1.0,
                  tol_days=0.02, verbose=True):
    """
    Per-row label, label_source ('manual', 'crossbin' or 'pipeline') and
    training weight, aligned with `features`.

    Manual labels are keyed on objid/eventid, which change whenever detection
    is re-run. A keyed match is only accepted if position and peak time agree
    (when the label carries them); otherwise, with rematch, the label is moved
    to the nearest event in the same cut and frame bin within tol_px / tol_days.

    The manual sort only shows one frame bin, but most events are also detected
    at coarser bins (linked through crossbin_ids). With crossbin_weight > 0 those
    detections of the same physical event inherit its manual label, so the model
    learns what each class looks like at every binning.
    """
    labels = pd.DataFrame({'label': pd.Series(np.nan, index=features.index, dtype=object),
                           'label_source': pd.Series(np.nan, index=features.index, dtype=object),
                           'weight': np.zeros(len(features))}, index=features.index)

    if pipeline_weight > 0 and 'classification' in features:
        m = features['classification'].isin(PIPELINE_CLASSES)
        labels.loc[m, 'label'] = features.loc[m, 'classification']
        labels.loc[m, 'label_source'] = 'pipeline'
        labels.loc[m, 'weight'] = pipeline_weight

    if manual is None or len(manual) == 0:
        return labels

    feats = features.reset_index()
    joined = feats.merge(manual, on=KEY_COLS, how='inner', suffixes=('', '_lab'))
    check = [c for c in ['xcentroid', 'ycentroid', 'mjd_max'] if f'{c}_lab' in joined and c in joined]
    bad = np.zeros(len(joined), bool)
    for c in check:
        diff = np.abs(joined[c] - joined[f'{c}_lab'])
        bad |= (diff > (tol_days if c == 'mjd_max' else tol_px)).to_numpy()
    if 'frame_bin_lab' in joined and 'frame_bin' in joined:
        bad |= (joined['frame_bin'] != joined['frame_bin_lab']).to_numpy() & joined['frame_bin_lab'].notna().to_numpy()
    matched = joined.loc[~bad, ['index', 'label'] + KEY_COLS]

    matched_keys = set(map(tuple, matched[KEY_COLS].to_numpy()))
    remaining = manual[[tuple(r) not in matched_keys for r in manual[KEY_COLS].to_numpy()]]

    moved = []
    can_move = {'xcentroid', 'ycentroid', 'mjd_max'} <= set(remaining) and {'xcentroid', 'ycentroid', 'mjd_max'} <= set(features)
    if rematch and can_move and len(remaining):
        for _, lab in remaining.iterrows():
            same = ((features.sector == lab.sector) & (features.camera == lab.camera) &
                    (features.ccd == lab.ccd) & (features.cut == lab.cut))
            if 'frame_bin' in features and np.isfinite(lab.get('frame_bin', np.nan)):
                same &= features.frame_bin == lab.frame_bin
            cand = features[same]
            if not len(cand):
                continue
            dist = np.hypot(cand.xcentroid - lab.xcentroid, cand.ycentroid - lab.ycentroid)
            ok = (dist <= tol_px) & (np.abs(cand.mjd_max - lab.mjd_max) <= tol_days)
            if ok.any():
                moved.append({'index': dist[ok].idxmin(), 'label': lab.label})
    assigned = pd.concat([matched[['index', 'label']], pd.DataFrame(moved, columns=['index', 'label'])])

    clash = assigned.duplicated('index', keep=False) & ~assigned.duplicated(['index', 'label'], keep=False)
    assigned = assigned[~clash].drop_duplicates('index')
    labels.loc[assigned['index'], 'label'] = assigned['label'].to_numpy()
    labels.loc[assigned['index'], 'label_source'] = 'manual'
    labels.loc[assigned['index'], 'weight'] = 1.0

    if crossbin_weight > 0 and 'crossbin_ids' in features:
        ids = features['crossbin_ids'].apply(_parse_ids)
        cuts = pd.Series(_group_keys(features), index=features.index)
        members = {}
        for idx, lst in ids.items():
            for i in lst:
                members.setdefault((cuts[idx], i), []).append(idx)
        inherited = {}
        for idx in labels.index[labels.label_source == 'manual']:
            for i in ids[idx]:
                for sibling in members.get((cuts[idx], i), []):
                    if labels.at[sibling, 'label_source'] != 'manual':
                        inherited.setdefault(sibling, set()).add(labels.at[idx, 'label'])
        agreed = {k: v.pop() for k, v in inherited.items() if len(v) == 1}
        if agreed:
            rows = list(agreed)
            labels.loc[rows, 'label'] = list(agreed.values())
            labels.loc[rows, 'label_source'] = 'crossbin'
            labels.loc[rows, 'weight'] = crossbin_weight

    if verbose:
        n_lost = len(manual) - len(assigned)
        print(f'Manual labels: {len(matched)} matched by key, {len(moved)} re-matched by position/time, '
              f'{n_lost} not found in the feature table.')
        for source, weight in [('crossbin', crossbin_weight), ('pipeline', pipeline_weight)]:
            if (labels.label_source == source).any():
                print(f'{source.capitalize()} labels: {(labels.label_source == source).sum()} (weight {weight}).')
    return labels


# ----------------------------- Model ----------------------------- #

def _group_keys(df, by='cut'):
    """Cross-validation groups: whole cuts, so no object (or its crossbin copies) spans train and test; or,
    with by='sector', whole sectors (tests how the model carries over to a sector it has never seen)."""
    if by == 'sector':
        return df['sector'].astype(str).to_numpy()
    return (df['sector'].astype(str) + '_' + df['camera'].astype(str) + '_' +
            df['ccd'].astype(str) + '_' + df['cut'].astype(str)).to_numpy()


def _aligned_proba(model, X, classes):
    """predict_proba with columns in `classes` order (classes unseen in training get ~0)."""
    p = np.full((len(X), len(classes)), 1e-6)
    raw = model.predict_proba(X)
    for j, c in enumerate(model.classes_):
        p[:, classes.index(c)] = raw[:, j]
    return p / p.sum(axis=1, keepdims=True)


def _fit_temperature(proba, y_idx, reg=1e-2):
    """
    Temperature plus per-class bias on log-probabilities, fitted by weighted
    log-loss. Fixes over/under-confidence and the prior shift from class
    balancing with only k parameters.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    logp = np.log(np.clip(proba, 1e-12, 1))
    k = logp.shape[1]
    rows = np.arange(len(y_idx))

    def loss(params):
        z = logp / np.exp(params[0]) + np.r_[0.0, params[1:]]
        z = z - logsumexp(z, axis=1, keepdims=True)
        return -np.mean(z[rows, y_idx]) + reg * np.sum(params[1:] ** 2)

    # temperature bounded to [0.5, 20]: on near-separable data the unbounded optimum sharpens
    # without limit, and a single confident miss in new data then costs arbitrarily much
    bounds = [(np.log(0.5), np.log(20))] + [(-5, 5)] * (k - 1)
    return minimize(loss, np.zeros(k), method='L-BFGS-B', bounds=bounds).x


def _apply_temperature(proba, params):
    from scipy.special import softmax

    logp = np.log(np.clip(proba, 1e-12, 1))
    return softmax(logp / np.exp(params[0]) + np.r_[0.0, params[1:]], axis=1)


def _log_loss(proba, y_idx):
    return -np.mean(np.log(np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1)))


def _no_flare_dips(proba, classes, flux_sign):
    """
    A flare is a brightening, so a dip (flux_sign < 0) can't be one: its Flare probability is shared out over the
    other classes in proportion. On S54 the model called 9 of 100 sampled Flare calls on dips; none were flares.
    """
    if flux_sign is None or 'Flare' not in classes:
        return proba
    proba = np.array(proba, dtype=float)
    dip = np.asarray(flux_sign, dtype=float) < 0
    if dip.any():
        proba[dip, list(classes).index('Flare')] = 0
        proba[dip] /= np.clip(proba[dip].sum(axis=1, keepdims=True), 1e-12, None)
    return proba


class EventClassifier():
    """
    Gradient-boosted classifier over the ml_classifier feature groups.

    feature_groups : which prefixes to use (subset of FEATURE_GROUPS).
    exclude : feature name prefixes to leave out; by default HOST_FEATURES, so
        the model can't use whether a star is there. () = use everything.
    class_balance : 0 = natural class frequencies, 1 = fully balanced;
        intermediate values up-weight rare classes without distorting the
        probabilities as much (the calibration step removes the remaining shift).
    """

    def __init__(self, feature_groups=FEATURE_GROUPS, exclude=HOST_FEATURES, class_balance=0.5, model_params=None,
                 random_state=0, cv_group='cut'):
        self.feature_groups = tuple(feature_groups)
        self.exclude = tuple(exclude)
        self.class_balance = class_balance
        self.model_params = {'learning_rate': 0.05, 'max_iter': 300, 'max_leaf_nodes': 31,
                             'min_samples_leaf': 20, 'l2_regularization': 1.0, 'early_stopping': False,
                             **(model_params or {})}
        self.random_state = random_state
        self.cv_group = cv_group            # 'cut' or 'sector': what a cross-validation fold holds out
        self.version = FEATURE_VERSION
        self.calibration_ = None

    def _new_model(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=self.random_state, **self.model_params)

    def _training_data(self, features, labels):
        lab = labels.reindex(features.index)
        m = lab['label'].notna().to_numpy()
        X = features.loc[m, self.feature_names_].to_numpy(float)
        y = lab.loc[m, 'label'].to_numpy(object)
        w = lab.loc[m, 'weight'].to_numpy(float)
        source = lab.loc[m, 'label_source'].to_numpy(object)

        # rarer classes up-weighted by (total / class weight) ** class_balance
        totals = pd.Series(w).groupby(y).sum()
        scale = (totals.sum() / (len(totals) * totals)) ** self.class_balance
        return m, X, y, w * scale.reindex(y).to_numpy(), source

    def cross_validate(self, features, labels, n_splits=5, importance=False, verbose=True):
        """
        Out-of-fold class probabilities for every labelled row, with folds made
        of whole cuts. With importance, also permutation importance (increase
        in log-loss on the held-out manual labels) per feature and per group.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        if not hasattr(self, 'feature_names_'):
            self.feature_names_ = _feature_columns(features.columns, self.feature_groups, self.exclude + LEAKY_FEATURES)
        m, X, y, w, source = self._training_data(features, labels)
        classes = [c for c in CLASSES if c in set(y)] + sorted(set(y) - set(CLASSES))
        groups = _group_keys(features.loc[m], getattr(self, 'cv_group', 'cut'))
        k = min(n_splits, len(np.unique(groups)))
        if k < 2:
            raise ValueError('Cross-validation needs labelled events from at least two cuts.')

        rng = np.random.default_rng(self.random_state)
        oof = np.full((len(y), len(classes)), np.nan)
        fold = np.full(len(y), -1)
        imp_rows = []
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        for i, (tr, te) in enumerate(splitter.split(X, y, groups)):
            model = self._new_model().fit(X[tr], y[tr], sample_weight=w[tr])
            oof[te] = _aligned_proba(model, X[te], classes)
            fold[te] = i
            if verbose:
                print(f'  fold {i + 1}/{k}: {len(tr)} train, {len(te)} test')
            if importance:
                test = te[(source[te] == 'manual') & np.isin(y[te], model.classes_)]
                if len(test) >= 10:
                    imp_rows += self._permutation_importance(model, X[test], y[test], classes, rng)

        out = features.loc[m, [c for c in META_COLS if c in features]].copy()
        for j, c in enumerate(classes):
            out[f'p_{c}'] = oof[:, j]
        out['label'] = y
        out['label_source'] = source
        out['fold'] = fold
        self.classes_ = classes
        if importance and imp_rows:
            imp = pd.DataFrame(imp_rows)
            self.importance_ = imp.groupby(['kind', 'name'])['delta_log_loss'].agg(['mean', 'std']).reset_index()
        return out

    def _permutation_importance(self, model, X, y, classes, rng, n_repeats=3):
        y_idx = np.array([classes.index(c) for c in y])
        base = _log_loss(_aligned_proba(model, X, classes), y_idx)
        rows = []
        groups = {g: [j for j, f in enumerate(self.feature_names_) if f.startswith(f'{g}_')] for g in self.feature_groups}
        targets = [('feature', f, [j]) for j, f in enumerate(self.feature_names_)]
        targets += [('group', g, cols) for g, cols in groups.items() if cols]
        for kind, name, cols in targets:
            deltas = []
            for _ in range(n_repeats):
                Xp = X.copy()
                perm = rng.permutation(len(X))
                Xp[:, cols] = X[perm][:, cols]
                deltas.append(_log_loss(_aligned_proba(model, Xp, classes), y_idx) - base)
            rows.append({'kind': kind, 'name': name, 'delta_log_loss': float(np.mean(deltas))})
        return rows

    def fit(self, features, labels, n_splits=5, calibrate=True, importance=False, verbose=True):
        """
        Fit on every labelled row. With calibrate (default), first runs grouped
        cross-validation: the out-of-fold predictions fit the calibration, and
        a cross-fitted calibrated copy is kept as `oof_` for honest evaluation.
        """
        self.feature_names_ = _feature_columns(features.columns, self.feature_groups, self.exclude + LEAKY_FEATURES)
        self.oof_ = None
        if calibrate:
            raw = self.cross_validate(features, labels, n_splits=n_splits, importance=importance, verbose=verbose)
            self.oof_raw_ = raw
            self.calibration_, self.oof_ = self._calibrate(raw)
            cols = [f'p_{c}' for c in self.classes_]
            self.oof_[cols] = _no_flare_dips(self.oof_[cols].to_numpy(), self.classes_, self.oof_.get('flux_sign'))

        m, X, y, w, source = self._training_data(features, labels)
        self.classes_ = [c for c in CLASSES if c in set(y)] + sorted(set(y) - set(CLASSES))
        self.model_ = self._new_model().fit(X, y, sample_weight=w)

        manual = X[source == 'manual']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ref = manual if len(manual) >= 20 else X
            self.domain_ = (np.nanquantile(ref, 0.005, axis=0), np.nanquantile(ref, 0.995, axis=0))
        self.training_summary_ = pd.crosstab(pd.Series(y, name='label'), pd.Series(source, name='source'))
        if verbose:
            print(self.training_summary_.to_string())
        return self

    def _calibrate(self, raw):
        """Temperature/bias calibration on manual out-of-fold rows; returns (params, cross-fitted oof)."""
        cols = [f'p_{c}' for c in self.classes_]
        manual = (raw['label_source'] == 'manual').to_numpy()
        if manual.sum() < 20:
            warnings.warn('Fewer than 20 manual labels: probabilities left uncalibrated.')
            return None, raw.copy()

        P = raw[cols].to_numpy()
        y_idx = np.array([self.classes_.index(c) for c in raw['label']])
        folds = raw['fold'].to_numpy()
        calibrated = P.copy()
        for f in np.unique(folds):
            train = manual & (folds != f)
            if train.sum() >= 20:
                calibrated[folds == f] = _apply_temperature(P[folds == f], _fit_temperature(P[train], y_idx[train]))
        out = raw.copy()
        out[cols] = calibrated
        return _fit_temperature(P[manual], y_idx[manual]), out

    def predict(self, features):
        """
        Calibrated class probabilities plus summary columns:
          p_astrophysical  total probability of a non-artefact class
          pred_class / pred_coarse
          entropy          normalised 0-1; high = the model is unsure (review first)
          margin           gap between the two most likely classes
          domain_out_frac  fraction of features outside the range of the manual
                           training labels -- high values mean the prediction
                           rests on weak labels or extrapolation
        """
        missing = [c for c in self.feature_names_ if c not in features]
        if missing:
            warnings.warn(f'{len(missing)} features missing from the input, treated as NaN: {missing[:5]}...')
        X = features.reindex(columns=self.feature_names_).to_numpy(float)
        proba = _aligned_proba(self.model_, X, self.classes_)
        if self.calibration_ is not None:
            proba = _apply_temperature(proba, self.calibration_)
        proba = _no_flare_dips(proba, self.classes_, features.get('flux_sign'))
        out = self._summarise(features, proba)
        lo, hi = self.domain_
        finite = np.isfinite(X)
        outside = ((X < lo) | (X > hi)) & finite
        out['domain_out_frac'] = outside.sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
        return out

    def _summarise(self, features, proba):
        out = features[[c for c in META_COLS if c in features]].copy()
        classes = np.array(self.classes_)
        for j, c in enumerate(classes):
            out[f'p_{c}'] = proba[:, j]
        astro = ~np.isin(classes, ARTEFACT_CLASSES)
        out['p_astrophysical'] = proba[:, astro].sum(axis=1)
        out['pred_class'] = classes[np.argmax(proba, axis=1)]
        out['pred_coarse'] = np.where(out['p_astrophysical'] >= 0.5, 'Astrophysical', 'Artefact')
        top2 = np.sort(proba, axis=1)[:, -2:]
        out['margin'] = top2[:, 1] - top2[:, 0]
        out['entropy'] = -np.sum(proba * np.log(np.clip(proba, 1e-12, 1)), axis=1) / np.log(len(classes))
        return out

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        import joblib
        obj = joblib.load(path)
        if getattr(obj, 'version', None) != FEATURE_VERSION:
            warnings.warn(f'Model built with feature version {getattr(obj, "version", None)}, '
                          f'current is {FEATURE_VERSION}: re-extract features or retrain.')
        # A model saved with scikit-learn 1.3 lacks the _preprocessor attribute that 1.4+ checks when predicting.
        # It is only set for categorical features (none here), so None = no preprocessing, as in 1.3. Checked:
        # identical probabilities (to 2e-16) on 9,975 S54 events, cluster sklearn vs local 1.3.2.
        model = getattr(obj, 'model_', None)
        if model is not None and not hasattr(model, '_preprocessor'):
            model._preprocessor = None
        return obj


# ----------------------------- Classifying a cut (the detector's stage 1) ----------------------------- #

_LOADED = {}


def load_default_classifier(path=None):
    """The classifier shipped with the package (ML_VERSION), loaded once per process."""
    path = path or DEFAULT_MODEL_PATH
    if path not in _LOADED:
        _LOADED[path] = EventClassifier.load(path)
    return _LOADED[path]


def classify_cut(data_path, sector, cam, ccd, cut, n=8, model=None, unsure_below=UNSURE_BELOW, config=None,
                 events=None, progress=False, n_jobs=1):
    """
    Stage-1 class probabilities for every event in a cut's detected_events.csv.

    The model is trained on frame-bin-1 events, so those are scored directly (features from the cut's flux cube,
    exactly as for training). An event in a coarser bin takes the probabilities of its frame-bin-1 crossbin sibling
    -- the same physical event -- choosing the most significant if there are several; one with no frame-bin-1
    sibling gets NaN probabilities.

    ml_classification is the most likely class, or 'Unsure' when that class's probability is below unsure_below
    (or there are no probabilities).

    events : the cut's event table if already in memory (the detector's, before saving); None = read
        detected_events.csv.
    n_jobs : joblib workers for the feature extraction (see extract_cut_features).

    Returns objid, eventid, p_<class> per class and ml_classification, one row per event in the table's order.
    """
    model = model or load_default_classifier()
    if events is None:
        events = load_cut_events(data_path, sector, cam, ccd, cut, n)
    else:
        events = _as_loaded(events)
    classes = list(model.classes_)
    p_cols = [f'p_{c}' for c in classes]
    out = events[['objid', 'eventid']].copy()
    for c in p_cols:
        out[c] = np.nan

    fb1 = (events['frame_bin'] == 1).to_numpy()
    if fb1.any():
        cfg = {'crossmatch': False, 'max_tagged': None, **(config or {})}
        feats = extract_cut_features(data_path, sector, cam, ccd, cut, n, events=events[fb1], config=cfg,
                                     all_events=events, progress=progress, n_jobs=n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pred = model.predict(feats)
        scored = out[['objid', 'eventid']].reset_index().merge(pred[['objid', 'eventid'] + p_cols],
                                                               on=['objid', 'eventid']).set_index('index')
        out.loc[scored.index, p_cols] = scored[p_cols].to_numpy()

    # -- Coarser bins inherit from their frame-bin-1 sibling -- #
    if 'crossbin_ids' in events and (~fb1).any():
        ids = events['crossbin_ids'].apply(_parse_ids)
        sig = pd.to_numeric(events.get('lc_sig_max', pd.Series(0.0, index=events.index)), errors='coerce').fillna(0)
        best = {}                          # crossbin id -> the most significant scored frame-bin-1 event
        for idx in events.index[fb1]:
            if out.loc[idx, p_cols].notna().all():
                for i in ids[idx]:
                    if i not in best or sig[idx] > sig[best[i]]:
                        best[i] = idx
        for idx in events.index[~fb1]:
            sources = [best[i] for i in ids[idx] if i in best]
            if sources:
                out.loc[idx, p_cols] = out.loc[max(sources, key=lambda j: sig[j]), p_cols].to_numpy()

    P = out[p_cols].to_numpy(float)
    ok = np.isfinite(P).all(axis=1)
    P0 = np.where(ok[:, None], P, 0.0)
    top, arg = P0.max(axis=1), np.array(classes)[P0.argmax(axis=1)]
    out['ml_classification'] = np.where(ok & (top >= unsure_below), arg, 'Unsure')
    return out


# ----------------------------- Evaluation / review ----------------------------- #

def evaluate(predictions, labels=None, source='manual'):
    """
    Metrics for predictions with known labels (e.g. clf.oof_), restricted to
    one label source. Returns a dict of DataFrames / values:
      per_class, confusion  -- fine classes
      coarse                -- artefact vs astrophysical: ROC AUC, average
                               precision, and the artefact rejection /
                               purity at fixed astrophysical recall
      calibration           -- log-loss, Brier score, expected calibration error
    `baseline_purity` is the astrophysical fraction of the evaluated set: for
    manually sorted events (which already passed filter_events) it is the
    purity of the current rule-based filtering.
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

    df = predictions.copy()
    if labels is not None:
        df[['label', 'label_source']] = labels.loc[df.index, ['label', 'label_source']]
    if source is not None:
        df = df[df['label_source'] == source]
    df = df[df['label'].notna()]
    classes = [c[2:] for c in df.columns if c.startswith('p_') and c != 'p_astrophysical']
    df = df[df['label'].isin(classes)]
    if len(df) == 0:
        raise ValueError('No labelled rows to evaluate.')

    P = df[[f'p_{c}' for c in classes]].to_numpy()
    y = df['label'].to_numpy()
    y_idx = np.array([classes.index(c) for c in y])
    pred = np.array(classes)[np.argmax(P, axis=1)]
    present = [c for c in classes if c in set(y) or c in set(pred)]

    result = {'n': len(df)}
    result['per_class'] = pd.DataFrame(classification_report(y, pred, labels=present, output_dict=True,
                                                             zero_division=0)).T
    result['confusion'] = pd.DataFrame(confusion_matrix(y, pred, labels=present), index=present, columns=present)

    astro_cols = [j for j, c in enumerate(classes) if c not in ARTEFACT_CLASSES]
    p_astro = P[:, astro_cols].sum(axis=1)
    is_astro = ~np.isin(y, ARTEFACT_CLASSES)
    result['baseline_purity'] = float(is_astro.mean())
    coarse = {}
    if 0 < is_astro.sum() < len(y):
        coarse['roc_auc'] = roc_auc_score(is_astro, p_astro)
        coarse['average_precision'] = average_precision_score(is_astro, p_astro)
        for recall in (0.9, 0.95, 0.99):
            thr = np.quantile(p_astro[is_astro], 1 - recall)
            keep = p_astro >= thr
            coarse[f'recall{int(recall * 100)}_threshold'] = float(thr)
            coarse[f'recall{int(recall * 100)}_artefacts_removed'] = float(np.mean(~keep[~is_astro]))
            coarse[f'recall{int(recall * 100)}_purity'] = float(np.mean(is_astro[keep]))
    result['coarse'] = coarse

    onehot = np.eye(len(classes))[y_idx]
    conf = P.max(axis=1)
    correct = pred == y
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    ece = sum(np.abs(conf[which == b].mean() - correct[which == b].mean()) * np.mean(which == b)
              for b in range(10) if np.any(which == b))
    result['calibration'] = {'log_loss': _log_loss(P, y_idx), 'brier': float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
                             'ece': float(ece)}
    return result


def rank_for_review(predictions, labels=None, n=None, by='entropy'):
    """
    Unlabelled events ordered by how much a manual label would teach the model:
    'entropy' (unsure between any classes) or 'boundary' (p_astrophysical
    closest to 0.5). Adds the plot_events / manual_sort image name.
    """
    df = predictions.copy()
    if labels is not None:
        df = df[~labels.reindex(df.index)['label_source'].isin(['manual', 'crossbin'])]
    score = df['entropy'] if by == 'entropy' else 1 - np.abs(2 * df['p_astrophysical'] - 1)
    df = df.assign(review_score=score).sort_values('review_score', ascending=False)
    df['image'] = [f'S{r.sector}C{r.camera}C{r.ccd}C{r.cut}O{r.objid}E{r.eventid}.png'
                   for r in df[KEY_COLS].itertuples(index=False)]
    return df.head(n) if n is not None else df

"""
Forced photometry, detrending, star-contamination flagging, and
shift-and-stack for the asteroid tracks predicted by asteroid_prediction.py.

Runs as a pipeline stage after reduce() (needs the reduced flux cube),
consuming a cut's own predicted ephemeris (positions already in cut-local
pixel coordinates) plus its already-cached local Gaia catalogue -- no new
network queries needed for either input.

Two independent photometry methods are computed per predicted position:
  - forced aperture photometry (simple, robust to PRF-model mismatch)
  - forced PSF photometry (matches the flux convention of tessreduce's own
    zeropoint calibration -- see psf_flux_calibration.py's _fit_star, a
    PRF scene-fit, not an aperture sum)
Both matter: aperture photometry is more forgiving of a poorly-matched PRF
near a bright star's halo, while PSF photometry is much better at rejecting
that same halo contamination once pixel-phase systematics are removed (see
detrend_pixel_phase) -- comparing the two is itself a useful diagnostic.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

import numpy as np
import pandas as pd

APERTURE_RADIUS_PX = 1.5
STAMP_SIZE = 9
GAIA_MAG_LIMIT = 19.0


# ---------------------------------------------------------------------------
# Shared-memory-backed parallel dispatch across tracks
# ---------------------------------------------------------------------------
#
# Forced photometry on a dense cut (hundreds of tracks) is easily the
# dominant cost of the asteroid_lightcurves stage, and was entirely
# single-threaded (confirmed: CPU usage stayed near one core regardless of
# cpus-per-task requested). Each track's own frames are independent of every
# other track's, so this parallelises cleanly across tracks. The cube is put
# in a multiprocessing.shared_memory block rather than pickled into each
# worker's initargs, so N workers don't each hold their own full copy of a
# (potentially large) reduced flux cube -- doubling as memory as well as CPU
# usage would defeat the point on exactly the dense, memory-tight cuts this
# is meant to help.

def _available_cpu_count():
    """os.cpu_count() reports the WHOLE NODE's core count, not this process's actual
    scheduling allocation -- on a shared cluster node that's a real, large difference (
    confirmed on ozstar: os.cpu_count()==36 vs the 8 cores an 8-cpus-per-task SLURM job was
    actually given). Sizing a worker pool off it oversubscribes massively: 36 processes
    competing for 8 real cores' worth of cgroup CPU time, each getting only a sliver
    (confirmed live: 36 worker processes sampled at ~2.8% CPU each, not the ~100%/8≈12.5%
    a correctly-sized pool would show). os.sched_getaffinity(0) reflects the actual cgroup/
    affinity-restricted core count and is what SLURM (or any cgroup-based scheduler) sets;
    it doesn't exist on macOS, so this falls back to cpu_count() there, which is fine since
    there's no cgroup restriction to respect on a local dev machine anyway."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


_worker_cube = None
_worker_shm = None
_worker_sigma_clip_cache = None
_worker_cpu_affinity = None


def _reset_worker_affinity():
    """Undo the affinity narrowing (see _photometry_worker_init) if it recurs. Confirmed
    live (via os.sched_getaffinity(0) bracketing a real _psf_lc_core call) that a forked
    worker's affinity is still correct going in, then collapses to a single CPU immediately
    after -- and stays that way for the rest of that worker's life, i.e. it's a one-time
    event per worker, not per call, but resetting defensively around every call is what
    guarantees it never matters which call triggers it or whether the env-var fix in
    _photometry_worker_init (set post-fork, so not certain to affect a BLAS thread pool
    that may have already initialized in the parent before fork) actually prevents it."""
    if _worker_cpu_affinity is not None:
        try:
            os.sched_setaffinity(0, _worker_cpu_affinity)
        except AttributeError:
            pass  # no sched_setaffinity (e.g. macOS)


def _photometry_worker_init(shm_name, shape, dtype, cpu_affinity=None):
    global _worker_cube, _worker_shm, _worker_sigma_clip_cache, _worker_cpu_affinity
    # A forked worker process was found, on ozstar, to have its own cpu affinity silently
    # narrowed down to a single core (confirmed live via taskset -pc: the main process kept
    # its full 8-cpu SLURM allocation, but every one of its forked children ended up pinned
    # to just one CPU) -- the classic OpenBLAS/MKL behaviour of a thread pool pinning itself
    # to one core on first use inside a forked process. This is why 8 workers were confirmed
    # (via /proc tick-sampling AND per-cpu mpstat) to collectively use only ~1 core's worth
    # of aggregate CPU despite all showing "R" (runnable) -- they were all actually
    # competing for time on that single shared core, not spread across the 8 real ones.
    #
    # Two defenses, since it isn't certain which (if either alone) is sufficient: force
    # single-threaded BLAS (parallelism here comes from the process pool, not from each
    # worker also multi-threading its own linear algebra, so this is correct regardless);
    # and keep the captured main-process affinity around so _reset_worker_affinity can be
    # called defensively around the actual call that was confirmed to trigger the
    # narrowing, since a one-time reset here wouldn't survive it recurring mid-pool-lifetime.
    for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_env] = "1"
    _worker_cpu_affinity = cpu_affinity
    _reset_worker_affinity()
    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_cube = np.ndarray(shape, dtype=dtype, buffer=_worker_shm.buf)
    # fresh per worker process, i.e. per cube (a new pool -- and so a new call to this
    # initializer -- is created for every forced_aperture_photometry call): a persistent
    # dict here, reused across every track task this worker is dispatched over the pool's
    # whole lifetime, is what makes the frame-level sigma-clip cache in
    # _forced_aperture_photometry_core actually pay off in the parallel path -- see there
    _worker_sigma_clip_cache = {}


def _aperture_track_worker(track_df, radius_px):
    return _forced_aperture_photometry_core(_worker_cube, track_df, radius_px,
                                              sigma_clip_cache=_worker_sigma_clip_cache)


def _psf_track_worker(track_df, sector, cam, ccd, ccd_x0, ccd_y0, prf_path, stamp_size):
    return _forced_psf_photometry_core(_worker_cube, track_df, sector, cam, ccd, ccd_x0, ccd_y0, prf_path, stamp_size)


def _run_parallel_by_track(cube, ephemeris_df, track_worker, worker_args, empty_columns, n_workers):
    """Shared dispatch: one task per track (designation), against a shared-memory copy of
    cube, via track_worker(track_df, *worker_args) run in each subprocess (must be a
    module-level function so ProcessPoolExecutor can pickle it). Falls back to sequential
    (no subprocess/shared-memory overhead) for a single track or n_workers<=1.

    Tracks are dispatched in bounded batches rather than submitting every track's future
    up front and holding every individual result DataFrame in memory simultaneously until
    one final all-at-once concat -- on an exceptionally dense cut (confirmed: one field
    with several thousand predicted tracks) that peak, not the shared cube itself, was
    what pushed jobs to OOM even at 224GB. A batch a few dozen tracks deep per worker keeps
    every worker fed without that unbounded peak; each batch's small set of results gets
    concatenated and freed before the next batch starts, so peak memory scales with batch
    size and worker count, not total track count."""
    if len(ephemeris_df) == 0:
        return pd.DataFrame(columns=empty_columns)

    designations = ephemeris_df["designation"].unique()
    if n_workers is None:
        n_workers = max(1, min(len(designations), _available_cpu_count()))

    if n_workers <= 1 or len(designations) <= 1:
        return None  # signal: caller should run its own sequential _core directly

    batch_size = max(n_workers * 20, 100)

    # Batches only close out once every one of their own futures resolves, so a batch's
    # own wall time is set by its single slowest track, not its average -- and track length
    # varies hugely on a real cut (confirmed: 1 to 3116 rows on one field, std/mean ~64%),
    # so batches built from designations' plain appearance order draw an arbitrary mix of
    # big and small tracks, and most workers finish their (smaller) track well before that
    # batch's biggest straggler, then sit idle until it's done. Sorting descending and
    # chunking CONTIGUOUSLY (not round-robin -- confirmed by simulation that round-robin
    # doesn't help at all, since it just reproduces the same overall size mix in every
    # batch) groups similarly-sized tracks together instead: low variance within a batch
    # means workers finish close together, so little idle waiting either way -- one batch
    # of uniformly big tracks that's genuinely slow throughout, not fast-but-blocked-on-one.
    # One groupby pass up front, not a fresh ephemeris_df["designation"] == d boolean scan
    # of the FULL frame for every single track -- that scan is O(len(ephemeris_df)) each,
    # and doing it once per track (1783 times on a real dense cut) made it O(n_tracks *
    # len(ephemeris_df)), ~3.3 BILLION comparisons total, entirely on the one main-process
    # core building each batch's futures -- confirmed live: sampling actual (not ps's
    # lifetime-averaged) CPU ticks mid-run showed the main process pinned at 100% while
    # every worker sat at ~2.7%, i.e. genuinely idle, not just imbalanced. Workers can never
    # be fed faster than the main process can hand them work, so this dwarfed the batch-
    # scheduling fix above -- that fix still matters (it's what determines whether the
    # WORKERS themselves idle on each other once fed), but this is what was actually
    # keeping them all starved.
    groups = dict(tuple(ephemeris_df.groupby("designation")))
    order = ephemeris_df.groupby("designation").size().sort_values(ascending=False).index.to_numpy()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    try:
        main_affinity = os.sched_getaffinity(0)
    except AttributeError:
        main_affinity = None  # e.g. macOS -- _photometry_worker_init handles None as a no-op

    shm = shared_memory.SharedMemory(create=True, size=cube.nbytes)
    try:
        shared_cube = np.ndarray(cube.shape, dtype=cube.dtype, buffer=shm.buf)
        shared_cube[:] = cube[:]
        batch_results = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_photometry_worker_init,
                                  initargs=(shm.name, cube.shape, cube.dtype, main_affinity)) as pool:
            for batch in batches:
                futures = [pool.submit(track_worker, groups[d], *worker_args)
                           for d in batch]
                results = [fut.result() for fut in as_completed(futures)]
                results = [r for r in results if len(r)]
                if results:
                    batch_results.append(pd.concat(results, ignore_index=True))
                del results
    finally:
        shm.close()
        shm.unlink()

    if not batch_results:
        return pd.DataFrame(columns=empty_columns)
    return pd.concat(batch_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Forced aperture photometry
# ---------------------------------------------------------------------------

_APERTURE_COLUMNS = ["designation", "frame", "mjd", "x", "y", "flux", "sky_std", "sig"]


def forced_aperture_photometry(cube, ephemeris_df, radius_px=APERTURE_RADIUS_PX, n_workers=None):
    """Forced circular-aperture photometry at every predicted (frame, x, y)
    in ephemeris_df, with local sky background from a concentric annulus
    (sigma-clipped to reject nearby real sources). x, y must already be in
    the cube's own local pixel frame and ephemeris_df must carry a 'frame'
    column indexing directly into cube's first axis (see
    match_ephemeris_to_reduced_frames for building this against the
    REDUCED cube's own frame list, which need not match the raw cut's).

    Parallelised across tracks (see _run_parallel_by_track) when there's
    more than one and n_workers != 1; pass n_workers=1 to force sequential."""
    result = _run_parallel_by_track(cube, ephemeris_df, _aperture_track_worker,
                                      (radius_px,), _APERTURE_COLUMNS, n_workers)
    if result is not None:
        return result
    return _forced_aperture_photometry_core(cube, ephemeris_df, radius_px)


def _forced_aperture_photometry_core(cube, ephemeris_df, radius_px, sigma_clip_cache=None):
    from astropy.stats import sigma_clip
    from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

    # sigma_clip(frame, ...) below depends only on which frame this row is, never on the
    # track's own position -- but with every track's rows sharing the same ~3000-ish
    # frames, the naive per-row call recomputes the identical whole-frame clip hundreds of
    # times over (confirmed: 1.83M rows against 3116 unique frames on a real dense cut,
    # ~588x redundant on average) and was the dominant cost of the whole asteroid_lightcurves
    # stage. Caching per frame index removes that redundancy; sigma_clip_cache is None (a
    # fresh dict, scoped to just this one call) in the sequential fallback path, where a
    # single call already covers every track's rows together, or the calling worker's own
    # persistent dict in the parallel path (see _photometry_worker_init) so the cache still
    # pays off there even though each worker call only ever sees one track's own rows.
    if sigma_clip_cache is None:
        sigma_clip_cache = {}

    r_in, r_out = 2 * radius_px + 3, 2 * radius_px + 7
    rows = []
    for row in ephemeris_df.itertuples():
        if not (0 <= row.x < cube.shape[2] and 0 <= row.y < cube.shape[1]):
            continue
        frame_idx = int(row.frame)
        frame = cube[frame_idx]
        aperture = CircularAperture([(row.x, row.y)], radius_px)
        annulus = CircularAnnulus([(row.x, row.y)], r_in, r_out)
        mask = sigma_clip_cache.get(frame_idx)
        if mask is None:
            mask = sigma_clip(frame, masked=True, sigma=5).mask
            sigma_clip_cache[frame_idx] = mask
        sky = ApertureStats(frame, annulus, mask=mask)
        phot = aperture_photometry(frame, aperture)
        npix = aperture.area
        sky_mean, sky_std = float(sky.mean[0]), float(sky.std[0])
        flux = float(phot["aperture_sum"].value[0]) - npix * sky_mean
        sig = flux / (np.sqrt(npix) * sky_std) if sky_std and np.isfinite(sky_std) and sky_std > 0 else np.nan
        rows.append(dict(designation=row.designation, frame=frame_idx, mjd=row.mjd,
                          x=row.x, y=row.y, flux=flux, sky_std=sky_std, sig=sig))
    # pd.DataFrame([]) has no columns at all (not even 'x') -- a fully out-of-bounds track
    # (e.g. every frame lands within STAMP_SIZE/2 of a cut's edge) would otherwise silently
    # break every downstream column access instead of just contributing zero rows
    if not rows:
        return pd.DataFrame(columns=_APERTURE_COLUMNS)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forced PSF photometry
# ---------------------------------------------------------------------------

_PSF_COLUMNS = ["designation", "frame", "mjd", "x", "y", "flux", "e_flux", "background", "sig"]


def forced_psf_photometry(cube, ephemeris_df, sector, cam, ccd, ccd_x0, ccd_y0,
                            prf_path=None, stamp_size=STAMP_SIZE, n_workers=None):
    """Forced PSF photometry (joint fit of [target PRF shape, flat local
    background]) at every predicted (frame, x, y), reusing the same
    per-frame linear solve psf_flux_calibration.py's zeropoint calibration
    uses (_psf_lc_core), so the resulting flux is directly comparable to
    the cut's own AB zeropoint without an aperture correction.

    ccd_x0, ccd_y0 : the cut's own corner offset (full-CCD pixels), needed
    only to look up the right PRF model for this part of the CCD -- x, y
    in ephemeris_df stay in the cube's local frame throughout.

    Parallelised across tracks (see _run_parallel_by_track) when there's
    more than one and n_workers != 1; pass n_workers=1 to force sequential."""
    from .psf_flux_calibration import PRF_PATH_DEFAULT
    prf_path = prf_path or PRF_PATH_DEFAULT

    result = _run_parallel_by_track(cube, ephemeris_df, _psf_track_worker,
                                      (sector, cam, ccd, ccd_x0, ccd_y0, prf_path, stamp_size),
                                      _PSF_COLUMNS, n_workers)
    if result is not None:
        return result
    return _forced_psf_photometry_core(cube, ephemeris_df, sector, cam, ccd, ccd_x0, ccd_y0, prf_path, stamp_size)


def _forced_psf_photometry_core(cube, ephemeris_df, sector, cam, ccd, ccd_x0, ccd_y0, prf_path, stamp_size):
    from .psf_flux_calibration import _psf_lc_core

    half = stamp_size // 2
    rows = []
    for row in ephemeris_df.itertuples():
        xi, yi = int(round(row.x)), int(round(row.y))
        if xi - half < 0 or yi - half < 0 or xi + half + 1 > cube.shape[2] or yi + half + 1 > cube.shape[1]:
            continue
        stamp = cube[row.frame:row.frame + 1, yi - half:yi + half + 1, xi - half:xi + half + 1]
        ccd_x, ccd_y = ccd_x0 + row.x, ccd_y0 + row.y
        try:
            out = _psf_lc_core(stamp, [row.mjd], row.x - xi, row.y - yi, ccd_x, ccd_y,
                                cam, ccd, sector, prf_path=prf_path, stamp_size=stamp_size)
        except Exception:
            continue
        # confirmed live (see _photometry_worker_init) that the first call into this
        # (PRF-loading, BLAS-backed) function is exactly what triggers the affinity
        # narrowing in a forked worker -- reset immediately after every call, not just
        # once, since it isn't certain the env-var fix alone prevents it from recurring
        _reset_worker_affinity()
        flux, e_flux, bg = float(out["flux_counts"][0]), float(out["e_flux_counts"][0]), float(out["background"][0])
        sig = flux / e_flux if e_flux and np.isfinite(e_flux) and e_flux > 0 else np.nan
        rows.append(dict(designation=row.designation, frame=int(row.frame), mjd=row.mjd,
                          x=row.x, y=row.y, flux=flux, e_flux=e_flux, background=bg, sig=sig))
    # pd.DataFrame([]) has no columns at all (not even 'x') -- a fully out-of-bounds track
    # (e.g. every frame lands within STAMP_SIZE/2 of a cut's edge) or one where every frame's
    # PRF fit raised would otherwise silently break every downstream column access
    if not rows:
        return pd.DataFrame(columns=_PSF_COLUMNS)
    return pd.DataFrame(rows)


def match_ephemeris_to_reduced_frames(ephemeris_df, reduced_mjd, tol_days=1e-4):
    """Re-derive each ephemeris row's 'frame' index against the REDUCED
    cube's own Times array, by nearest mjd -- the raw cut TPF's frame list
    (what predict_asteroids used) and the reduced product's frame list
    (tessreduce can drop bad-quality frames during reduction) aren't
    guaranteed to match 1:1, so index correspondence can't be assumed.
    Rows with no reduced frame within tol_days are dropped (that exposure
    didn't survive reduction)."""
    reduced_mjd = np.asarray(reduced_mjd)
    order = np.argsort(reduced_mjd)
    sorted_mjd = reduced_mjd[order]
    idx = np.searchsorted(sorted_mjd, ephemeris_df["mjd"].values)
    idx = np.clip(idx, 1, len(sorted_mjd) - 1)
    left, right = sorted_mjd[idx - 1], sorted_mjd[idx]
    use_left = np.abs(ephemeris_df["mjd"].values - left) <= np.abs(right - ephemeris_df["mjd"].values)
    nearest_sorted_idx = np.where(use_left, idx - 1, idx)
    nearest_mjd = sorted_mjd[nearest_sorted_idx]
    within_tol = np.abs(nearest_mjd - ephemeris_df["mjd"].values) <= tol_days

    out = ephemeris_df[within_tol].copy()
    out["frame"] = order[nearest_sorted_idx[within_tol]]
    return out


# ---------------------------------------------------------------------------
# Pixel-phase detrending
# ---------------------------------------------------------------------------

MIN_POINTS_FOR_DETREND = 20
DETREND_SIGMA_CLIP = 3.0
DETREND_N_ITER = 4


def _detrend_design(xfrac, yfrac):
    return np.column_stack([np.ones_like(xfrac), xfrac, yfrac, xfrac * yfrac, xfrac ** 2, yfrac ** 2])


def _detrend_robust_fit(flux, xfrac, yfrac):
    """Fit flux against a full 2nd-order polynomial in sub-pixel phase
    (xfrac, yfrac) -- a bilinear-only model can't represent a radially
    symmetric intra-pixel sensitivity dip, which is what's actually
    observed (flux depends on distance from the nearest pixel centre, not
    x/y direction). Sigma-clipped so a track's own real signal isn't
    absorbed into the phase model."""
    mask = np.isfinite(flux)
    coeffs = None
    for _ in range(DETREND_N_ITER):
        A = _detrend_design(xfrac[mask], yfrac[mask])
        coeffs, *_ = np.linalg.lstsq(A, flux[mask], rcond=None)
        resid = flux[mask] - A @ coeffs
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma <= 0 or not np.isfinite(sigma):
            break
        keep = np.abs(resid - np.median(resid)) < DETREND_SIGMA_CLIP * sigma
        new_mask = mask.copy()
        new_mask[mask] = keep
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    return _detrend_design(xfrac, yfrac) @ coeffs


def detrend_pixel_phase(df, flux_col="flux", e_flux_col="e_flux"):
    """Per-track pixel-phase detrending: removes the modulation (not the
    mean level) of a 2nd-order polynomial fit of flux vs sub-pixel phase.
    Confirmed empirically on real TESS forced-PSF photometry that this is
    a real, non-negligible effect (~30% scatter reduction on a bright,
    otherwise-clean track) -- intra-pixel sensitivity variation that a
    correctly-positioned PRF model alone does not capture. Adds
    flux_detrended / sig_detrended columns; tracks with fewer than
    MIN_POINTS_FOR_DETREND rows are passed through unchanged."""
    out = df.copy()
    out["xfrac"] = out["x"] - np.round(out["x"])
    out["yfrac"] = out["y"] - np.round(out["y"])
    out["flux_detrended"] = out[flux_col]

    for designation, idx in out.groupby("designation").groups.items():
        idx = np.asarray(idx)
        if len(idx) < MIN_POINTS_FOR_DETREND:
            continue
        flux = out.loc[idx, flux_col].values
        model = _detrend_robust_fit(flux, out.loc[idx, "xfrac"].values, out.loc[idx, "yfrac"].values)
        correction = model - np.nanmean(model)
        out.loc[idx, "flux_detrended"] = flux - correction

    if e_flux_col in out.columns:
        out["sig_detrended"] = np.where(out[e_flux_col] > 0, out["flux_detrended"] / out[e_flux_col], np.nan)
    return out


# ---------------------------------------------------------------------------
# Star-contamination flagging
# ---------------------------------------------------------------------------

GMAG_REF = 10.0
RADIUS_REF_PX = 4.0
RADIUS_MIN_PX = 2.0
RADIUS_MAX_PX = 20.0
LOCAL_EXCESS_WINDOW = 151
LOCAL_EXCESS_NSIGMA = 3.0
MIN_PROXIMITY_RUN_LENGTH = 15


def flag_radius_px(mag):
    """Brightness-scaled proximity radius (~sqrt(flux) law, anchored at a
    genuinely bright-star threshold) -- validated directionally against
    several real contamination cases at increasing distance/faintness."""
    r = RADIUS_REF_PX * 10 ** (-0.2 * (mag - GMAG_REF))
    return np.clip(r, RADIUS_MIN_PX, RADIUS_MAX_PX)


def local_gaia_cat_to_stars(gaia_cat, wcs, ccd_x0, ccd_y0):
    """Converts the cut's already-cached local_gaia_cat.csv (ra, dec, Gmag,
    Source -- see catalog_queries.create_external_gaia_cat) into cut-local
    pixel positions, avoiding a fresh Gaia query for star flagging. Drops
    any star with a NaN Gmag (real Gaia data: ~0.4% of a typical cut's
    catalog) -- flag_star_contamination can't score an unratable star's
    exclusion radius anyway, and leaving it in poisons its vectorized
    argmin (a NaN flag_radius_px propagates into every distance-margin
    comparison, so argmin can get stuck returning that star's index
    instead of the true closest/worst one -- confirmed: this silently
    hid a real, obvious 0.55px/Gmag=9.76 contaminating star)."""
    gaia_cat = gaia_cat.dropna(subset=["Gmag"])
    x, y = wcs.all_world2pix(gaia_cat["ra"].values, gaia_cat["dec"].values, 0)
    return pd.DataFrame({"x": x - ccd_x0, "y": y - ccd_y0, "mag": gaia_cat["Gmag"].values})


def _local_excess_mask(mjd, flux):
    """A point only counts as contaminated if it's a genuine local outlier
    -- proximity to a star alone triggers on plain coincidence in a dense
    field even for a clean, bright, real track (confirmed: ~23% of one
    real object's frames pass near *some* star purely by chance with zero
    photometric effect, since its own signal is periodic and revisits many
    phases/positions). The window must be well wider than any real
    contamination event's duration (~15-64 frames observed) or the rolling
    median gets pulled up by the event itself, hiding it as its own
    baseline."""
    order = np.argsort(mjd)
    s_sorted = pd.Series(flux[order])
    baseline = s_sorted.rolling(LOCAL_EXCESS_WINDOW, center=True, min_periods=5).median()
    resid = s_sorted - baseline
    mad = (resid - resid.rolling(LOCAL_EXCESS_WINDOW, center=True, min_periods=5).median()).abs().rolling(
        LOCAL_EXCESS_WINDOW, center=True, min_periods=5).median() * 1.4826
    excess = resid.abs() > LOCAL_EXCESS_NSIGMA * mad
    excess_orig_order = np.empty(len(flux), dtype=bool)
    excess_orig_order[order] = excess.values
    return excess_orig_order


def _expand_flag_over_excess_runs(mjd, local_excess, proximity):
    """A real contamination event is one continuous physical crossing --
    the object's distance to the causing star varies smoothly over the
    event, so the strict per-point proximity radius can legitimately be
    satisfied for only part of a contiguous excess run. Once any point in
    a run has confirmed proximity, the whole run is flagged rather than
    only the sub-part that individually clears the radius."""
    order = np.argsort(mjd)
    excess_sorted = local_excess[order]
    prox_sorted = proximity[order]
    flagged_sorted = np.zeros(len(mjd), dtype=bool)

    run_start = None
    for i in range(len(excess_sorted) + 1):
        in_run = i < len(excess_sorted) and excess_sorted[i]
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            if prox_sorted[run_start:i].any():
                flagged_sorted[run_start:i] = True
            run_start = None

    flagged = np.empty(len(mjd), dtype=bool)
    flagged[order] = flagged_sorted
    return flagged


def _long_proximity_runs_mask(mjd, proximity, star_idx, min_run_length=MIN_PROXIMITY_RUN_LENGTH):
    """Catches contamination events too long for _local_excess_mask to see --
    once a contiguous proximity run spans a large enough fraction of
    LOCAL_EXCESS_WINDOW, the rolling-median baseline gets pulled up by the
    event itself and local_excess never fires for most of it (confirmed on
    a real 85+ frame event, one fixed Gmag=8.0 star throughout).

    Critically, a run only counts if it is continuous proximity to the SAME
    cataloged star (star_idx constant throughout) -- near_star_proximity
    alone is frequently true almost continuously in dense fields simply
    because some star or other is always within RADIUS_MAX_PX, as the
    object sweeps past many different field stars in quick succession
    (confirmed: one real track had near_star_proximity=True 85% of the time
    while its nearest-star magnitude varied from 4th to 22nd mag -- normal
    field density, not contamination). Requiring one fixed star_idx for the
    whole run is what distinguishes a genuine sustained contaminator from
    ordinary dense-field crossing."""
    order = np.argsort(mjd)
    prox_sorted = proximity[order]
    idx_sorted = star_idx[order]
    n = len(mjd)
    flagged_sorted = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not prox_sorted[i]:
            i += 1
            continue
        this_star = idx_sorted[i]
        j = i
        while j < n and prox_sorted[j] and idx_sorted[j] == this_star:
            j += 1
        if j - i >= min_run_length:
            flagged_sorted[i:j] = True
        i = j

    flagged = np.empty(n, dtype=bool)
    flagged[order] = flagged_sorted
    return flagged


def flag_star_contamination(df, stars, flux_col="flux"):
    """near_bright_star is True when a cataloged star falls within its own
    brightness-scaled radius (proximity) AND either the light curve shows a
    genuine local excess there (see _local_excess_mask), or the proximity
    itself is sustained long enough to be implausible as chance coincidence
    (see _long_proximity_runs_mask) -- proximity alone triggers on plain
    coincidence in a dense field, but a long sustained run of it does not
    (see _long_proximity_runs_mask's docstring). Adds
    contaminating_star_dist_px / _mag and near_bright_star columns."""
    out = df.copy()
    contaminating_star_idx = np.full(len(out), -1, dtype=int)
    if len(stars) == 0:
        # no known stars for this cut (e.g. no local_gaia_cat.csv) -- nothing can be flagged as
        # near one, but the KDTree query below would otherwise raise on zero input points
        out["near_star_proximity"] = False
        out["contaminating_star_dist_px"] = np.nan
        out["contaminating_star_mag"] = np.nan
    else:
        # A full (n_rows x n_stars) pairwise distance matrix was the actual dominant cost of
        # this whole stage on a dense cut -- e.g. ~180,000 photometry rows x several thousand
        # local stars is a many-GB temporary array, computed on a single core (this function
        # isn't parallelised) 4 times per job (aperture + PSF, x2 for the position-corrected
        # second pass). A star can only ever contaminate (negative margin = dist - radius)
        # within its own radius, capped at RADIUS_MAX_PX, so a KDTree query for stars within
        # that radius of each point needs only the handful actually nearby -- no dense matrix,
        # and no candidates at all (common for most points) skips the row entirely.
        from scipy.spatial import cKDTree

        sx, sy, smag = stars["x"].values, stars["y"].values, stars["mag"].values
        sr = flag_radius_px(smag)
        points = np.column_stack([out["x"].values, out["y"].values])

        near_star_proximity = np.zeros(len(out), dtype=bool)
        contaminating_star_dist_px = np.full(len(out), np.nan)
        contaminating_star_mag = np.full(len(out), np.nan)
        best_margin = np.full(len(out), np.inf)

        # query_ball_point's per-point candidate list scales with LOCAL STAR DENSITY, not a
        # fixed size -- fine at a normal cut's density, but confirmed live on one cut whose
        # local star catalog held 2.5 million Gaia sources (~35 stars/px^2): most rows' full
        # within-radius candidate list ran into the tens of thousands of entries each, and
        # even querying in row-chunks still built each chunk's full lists before any were
        # consumed, driving measured RSS past 36GB and still climbing regardless of chunk
        # size. A single fixed-k nearest-neighbour query was tried and reverted: it isn't
        # correctness-preserving in a field this dense -- a bright, large-radius star at a
        # moderate distance can genuinely have a worse (more negative) margin than dozens of
        # much closer but small-radius stars, so it can rank outside any fixed k by raw
        # distance and get silently missed.
        #
        # Instead, bin stars by their OWN flag radius into tiers and search each tier only
        # out to that tier's own max radius, against only that tier's stars -- provably
        # correct (every star gets a properly radius-bounded check against its real sr,
        # nothing is approximated) while still bounding memory: brighter/large-radius tiers
        # search a bigger area but hold far fewer stars (the luminosity function), and the
        # numerous faint tier (most real Gaia catalogs sit near GAIA_MAG_LIMIT, i.e. at or
        # near RADIUS_MIN_PX) only needs a small search radius, cutting its candidate density
        # by orders of magnitude versus searching every star out to RADIUS_MAX_PX.
        chunk_size = 20000
        tier_edges = np.geomspace(RADIUS_MIN_PX, RADIUS_MAX_PX, 7)
        for lo, hi in zip(tier_edges[:-1], tier_edges[1:]):
            tier_mask = (sr >= lo) & (sr <= hi) if lo == tier_edges[0] else (sr > lo) & (sr <= hi)
            if not tier_mask.any():
                continue
            tier_global_idx = np.nonzero(tier_mask)[0]
            tier_sx, tier_sy = sx[tier_mask], sy[tier_mask]
            tier_smag, tier_sr = smag[tier_mask], sr[tier_mask]
            tier_tree = cKDTree(np.column_stack([tier_sx, tier_sy]))

            for start in range(0, len(points), chunk_size):
                end = min(start + chunk_size, len(points))
                # sparse_distance_matrix instead of query_ball_point: the latter builds one
                # Python list per query point (confirmed via memray on a real dense cut to be
                # this module's dominant allocator by a wide margin -- ~166M allocations in
                # one job, all from list construction, not from anything done with the
                # candidates afterward). sparse_distance_matrix returns the identical set of
                # (star, point, distance) triples as one flat sparse array instead -- verified
                # matching pair counts and preserving exact-zero-distance pairs (sparse
                # formats can silently drop explicit zeros; this one doesn't) -- and gives
                # distances directly, so the separate np.hypot pass below is gone too.
                point_tree = cKDTree(points[start:end])
                sdm = tier_tree.sparse_distance_matrix(point_tree, max_distance=hi, output_type='coo_matrix')
                if sdm.nnz == 0:
                    continue
                row_idx = sdm.col + start
                col_idx = sdm.row
                d_all = sdm.data
                margin_all = d_all - tier_sr[col_idx]

                # best (minimum-margin) candidate per row -- pandas' groupby/idxmin picks
                # the first occurrence on ties, matching np.argmin's own tie-break, and is
                # well-tested rather than a hand-rolled scatter-reduction here
                best_pos = pd.Series(margin_all).groupby(row_idx).idxmin()
                rows_with_cand = best_pos.index.values
                pair_pos = best_pos.values

                cand_margin = margin_all[pair_pos]
                improved = cand_margin < best_margin[rows_with_cand]
                upd_rows = rows_with_cand[improved]
                upd_pos = pair_pos[improved]

                best_margin[upd_rows] = margin_all[upd_pos]
                contaminating_star_dist_px[upd_rows] = d_all[upd_pos]
                contaminating_star_mag[upd_rows] = tier_smag[col_idx[upd_pos]]
                contaminating_star_idx[upd_rows] = tier_global_idx[col_idx[upd_pos]]
                near_star_proximity[upd_rows] = margin_all[upd_pos] < 0

        out["near_star_proximity"] = near_star_proximity
        out["contaminating_star_dist_px"] = contaminating_star_dist_px
        out["contaminating_star_mag"] = contaminating_star_mag

    out["local_flux_excess"] = False
    out["near_bright_star"] = False
    contaminating_star_idx = pd.Series(contaminating_star_idx, index=out.index)
    for designation, idx in out.groupby("designation").groups.items():
        idx = np.asarray(idx)
        mjd_g = out.loc[idx, "mjd"].values
        excess_g = _local_excess_mask(mjd_g, out.loc[idx, flux_col].values)
        prox_g = out.loc[idx, "near_star_proximity"].values
        star_idx_g = contaminating_star_idx.loc[idx].values
        out.loc[idx, "local_flux_excess"] = excess_g
        out.loc[idx, "near_bright_star"] = (_expand_flag_over_excess_runs(mjd_g, excess_g, prox_g)
                                             | _long_proximity_runs_mask(mjd_g, prox_g, star_idx_g))

    return out


# ---------------------------------------------------------------------------
# Shift-and-stack
# ---------------------------------------------------------------------------

STACK_SIG_TARGET = 5.0


def _required_n(avg_sig, sig_target):
    if avg_sig is None or not np.isfinite(avg_sig) or avg_sig <= 0:
        return None
    return max(int(np.ceil((sig_target / avg_sig) ** 2)), 1)


def _stack_track(track, n_stack):
    track = track.sort_values("mjd").reset_index(drop=True)
    rows = []
    for bin_index, start in enumerate(range(0, len(track), n_stack)):
        chunk = track.iloc[start:start + n_stack]
        w = 1.0 / chunk["e_flux"].values ** 2
        flux = np.sum(chunk["flux_detrended"].values * w) / np.sum(w)
        e_flux = 1.0 / np.sqrt(np.sum(w))
        mjd_lo, mjd_hi = chunk["mjd"].min(), chunk["mjd"].max()
        rows.append(dict(mjd=chunk["mjd"].mean(), mjd_err=(mjd_hi - mjd_lo) / 2,
                          flux=flux, e_flux=e_flux, sig=flux / e_flux, n_frames=len(chunk),
                          bin_index=bin_index))
    return pd.DataFrame(rows)


def stack_lightcurves(psf_df, sig_target=STACK_SIG_TARGET):
    """Shift-and-stack for tracks not already detected on average. The
    'shift' is done by construction -- every frame's forced PSF flux is
    already measured at that frame's own predicted (moving) position, so
    per-frame measurements are already registered to the object's rest
    frame. What's left is combining enough consecutive (inverse-variance
    weighted) frames to beat the noise down by ~sqrt(N) until the combined
    significance clears sig_target. Frames flagged near_bright_star are
    excluded first -- contaminated frames shouldn't be co-added into a
    detection."""
    clean = psf_df[~psf_df["near_bright_star"]]
    summary_rows = []
    stacked_all = []
    for designation, track in clean.groupby("designation"):
        avg_sig = track["sig_detrended"].mean()
        if avg_sig > sig_target:
            summary_rows.append(dict(designation=designation, avg_sig=avg_sig,
                                      stacking_needed=False, n_stack=1,
                                      n_frames=len(track), achieved_sig=avg_sig))
            continue

        n_needed = _required_n(avg_sig, sig_target)
        if n_needed is None or n_needed > len(track):
            summary_rows.append(dict(designation=designation, avg_sig=avg_sig,
                                      stacking_needed=True, n_stack=None,
                                      n_frames=len(track), achieved_sig=np.nan))
            continue

        stacked = _stack_track(track, n_needed)
        stacked["designation"] = designation
        stacked_all.append(stacked)
        summary_rows.append(dict(designation=designation, avg_sig=avg_sig,
                                  stacking_needed=True, n_stack=n_needed,
                                  n_frames=len(track), achieved_sig=stacked["sig"].max()))

    if summary_rows:
        summary = pd.DataFrame(summary_rows).set_index("designation")
    else:
        summary = pd.DataFrame(columns=["designation", "avg_sig", "stacking_needed", "n_stack",
                                          "n_frames", "achieved_sig"]).set_index("designation")
    stacked_df = pd.concat(stacked_all, ignore_index=True) if stacked_all else pd.DataFrame(
        columns=["mjd", "mjd_err", "flux", "e_flux", "sig", "n_frames", "bin_index", "designation"])
    return summary, stacked_df


# ---------------------------------------------------------------------------
# Predicted-vs-measured offset from shift-and-stack images
# ---------------------------------------------------------------------------

STACK_CENTROID_SIG_THRESHOLD = 8.0
STACK_CENTROID_STAMP_HALF = 6
STACK_CENTROID_WINDOW = 2
STACK_CENTROID_MIN_STAMPS = 5


def measure_stack_centroid_offset(track, cube, sector, cam, ccd, ccd_x0, ccd_y0, prf_path=None,
                                    half=STACK_CENTROID_STAMP_HALF, min_stamps=STACK_CENTROID_MIN_STAMPS,
                                    search_radius_px=2.0):
    """Predicted-vs-measured offset for one track, from its own shift-and-
    stack image. Every frame's stamp is extracted at the predicted position
    (already recorded per-frame in track's x, y, frame columns) and
    sub-pixel shifted so the predicted position lands exactly at the stamp
    centre in every frame; averaging then gives a much higher-SNR image
    than any single frame, on which a real offset shows up as a
    displacement of the source from that centre.

    The offset is measured by fitting the actual calibrated TESS PRF
    template (the same model forced_psf_photometry uses for flux, via
    psf_flux_calibration's PRF.locate) against the stacked image at a grid
    of candidate sub-pixel offsets, picking whichever gives the best
    (least-squares) fit -- not a naive flux-weighted centroid. TESS's PRF
    is measurably asymmetric, so a real source displaced from centre isn't
    just "brighter on one side": matching the actual PRF shape at each
    candidate offset is a materially better position estimate than
    treating the local flux distribution as if it had no assumed shape.
    Seeded from the flux-weighted centroid (cheap, and a robust fallback if
    the PRF search fails to find anything better) as the starting guess.

    The same joint fit that finds the offset (template amplitude + flat
    background, solved by linear least-squares at the best-fit position)
    also yields a flux estimate and, from the fit residuals, its
    uncertainty -- the identical error convention psf_flux_calibration's
    _psf_lc_core uses (1.4826 * median absolute residual). That fitted
    significance is returned alongside the offset, so a caller can decide
    whether to trust THIS track's own offset without needing a separate
    photometry pass to measure SNR first -- unlike a per-frame forced
    photometry run, which this function never touches at all (only raw
    cube stamps, extracted directly from the predicted ephemeris
    positions).

    Returns (offset_x, offset_y, sig, flux, n_stamps) -- offset/sig/flux
    are NaN if fewer than min_stamps frames had a usable in-bounds stamp,
    or if the stack has no usable signal at all. flux is the fitted PRF
    template amplitude (raw counts) at the adopted offset when the PRF fit
    ran, or the background-subtracted window flux sum as a cruder fallback
    when it couldn't -- either way, the same quantity that sig is derived
    from, so a caller can convert it to a magnitude (with a known
    zeropoint) for a direct sanity check against the ephemeris's own
    predicted mag_expected."""
    from scipy.ndimage import shift as ndshift
    from .psf_flux_calibration import PRF_PATH_DEFAULT, _get_prf
    prf_path = prf_path or PRF_PATH_DEFAULT

    stamps = []
    for row in track.itertuples():
        xi, yi = int(round(row.x)), int(round(row.y))
        if xi - half < 0 or yi - half < 0 or xi + half + 1 > cube.shape[2] or yi + half + 1 > cube.shape[1]:
            continue
        stamp = cube[row.frame, yi - half:yi + half + 1, xi - half:xi + half + 1]
        dx, dy = row.x - xi, row.y - yi
        stamps.append(ndshift(stamp, (-dy, -dx), order=1, mode="nearest"))

    if len(stamps) < min_stamps:
        return np.nan, np.nan, np.nan, np.nan, len(stamps)

    # median, not mean -- a plain mean lets a handful of contaminated frames (cosmic rays,
    # background artifacts) among possibly hundreds bias the stacked image's apparent shape;
    # confirmed visually on a real track where the mean-stacked image showed a spurious
    # secondary feature the PRF search was chasing, absent from the median-stacked version
    stacked_mean = np.median(stamps, axis=0)
    stamp_size = 2 * half + 1

    # flux-weighted centroid in a small central window -- initial guess for the PRF
    # search below, and the fallback if that search can't be run/doesn't improve on it
    window = STACK_CENTROID_WINDOW
    bg = np.median(stacked_mean)
    sub = stacked_mean[half - window:half + window + 1, half - window:half + window + 1]
    yy, xx = np.mgrid[half - window:half + window + 1, half - window:half + window + 1]
    w = np.clip(sub - bg, 0, None)
    if w.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan, len(stamps)
    cx0 = (xx * w).sum() / w.sum() - half
    cy0 = (yy * w).sum() / w.sum() - half
    # crude fallback significance/flux (no PRF shape assumed) -- used only if the PRF fit
    # below can't run at all; a background-subtracted window sum over the robust
    # pixel-to-pixel scatter, the same style of estimator _psf_lc_core uses for its own
    # uncertainty
    flux0 = float(w.sum())
    noise = 1.4826 * np.median(np.abs(stacked_mean - bg))
    sig0 = float(flux0 / (noise * np.sqrt(w.size))) if noise > 0 else np.nan

    data = stacked_mean.ravel()
    finite = np.isfinite(data)
    if finite.sum() < 3:
        return cx0, cy0, sig0, flux0, len(stamps)

    try:
        last = track.iloc[-1]
        ccd_x, ccd_y = ccd_x0 + last.x, ccd_y0 + last.y
        prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
        prf = _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir)
    except Exception as ex:
        # falls back to the flux-weighted centroid, but not silently -- a wrong prf_path or
        # any other PRF-loading failure would otherwise degrade to the old behaviour with
        # no visible sign anything was wrong (caught a real instance of exactly this during
        # development: a stale default prf_path made every single call fail this way)
        import warnings
        warnings.warn(f"measure_stack_centroid_offset: PRF unavailable ({ex!r}), "
                       f"falling back to flux-weighted centroid", RuntimeWarning)
        return cx0, cy0, sig0, flux0, len(stamps)

    def _fit_at(dx, dy):
        """Joint [template amplitude, flat background] linear fit at one candidate
        offset. Returns (sse, flux, e_flux) -- e_flux/flux are None if the template
        itself is degenerate at this offset (kept out of the search via inf sse)."""
        template = prf.locate(half + dx, half + dy, (stamp_size, stamp_size))
        s = np.nansum(template)
        if not (np.isfinite(s) and s > 0):
            return np.inf, None, None
        template = (template / s).ravel()[finite]
        A = np.column_stack([template, np.ones(finite.sum())])
        d = data[finite]
        coeffs, *_ = np.linalg.lstsq(A, d, rcond=None)
        resid = d - A @ coeffs
        sse = float(np.sum(resid ** 2))
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        try:
            ATA_inv = np.linalg.inv(A.T @ A)
            e_flux = sigma * np.sqrt(ATA_inv[0, 0])
        except np.linalg.LinAlgError:
            e_flux = np.nan
        return sse, float(coeffs[0]), float(e_flux)

    from scipy.optimize import minimize
    # bounds=, not a manual np.inf wall outside the box -- an inf-return gives Nelder-Mead's
    # simplex a flat cliff instead of a real constraint, and it was confirmed pinning
    # multiple real tracks' fits exactly at the wall (e.g. (2.0, 2.0)) rather than finding
    # a genuine interior optimum. scipy's own bounds handling reflects/clips the simplex
    # properly instead.
    bounds = [(-search_radius_px, search_radius_px)] * 2
    res = minimize(lambda off: _fit_at(*off)[0], x0=[cx0, cy0], method="Nelder-Mead",
                    bounds=bounds, options={"xatol": 0.01, "fatol": 1e-6})

    sse_fit, flux_fit, e_flux_fit = _fit_at(*res.x)
    sse_naive, flux_naive, e_flux_naive = _fit_at(cx0, cy0)
    ox, oy, flux, e_flux = ((res.x[0], res.x[1], flux_fit, e_flux_fit) if res.success and sse_fit < sse_naive
                            else (cx0, cy0, flux_naive, e_flux_naive))
    # clamped at 0, not left negative -- the fitted amplitude can come out negative for a
    # noise-dominated fit (no real signal, not "negatively significant"), and a negative sig
    # is meaningless as a trust measure -- 0 is the correct floor, not a value to discard
    sig = max(0.0, float(flux / e_flux)) if (e_flux is not None and np.isfinite(e_flux) and e_flux > 0) else sig0
    return float(ox), float(oy), sig, float(flux), len(stamps)


def pool_offset_from_stacks(ephemeris, cube, sector, cam, ccd, ccd_x0, ccd_y0, prf_path=None, zp_ab=None,
                              sig_threshold=STACK_CENTROID_SIG_THRESHOLD, min_tracks=2, **stamp_kwargs):
    """Robust per-cut (dx, dy) offset, pooled (median) from the individual
    shift-and-stack centroid offsets (measure_stack_centroid_offset) of
    every track that reaches at least sig_threshold in ITS OWN joint
    position+flux fit -- these are the only tracks whose position is
    measured precisely enough to trust individually, unlike a single
    detected-source centroid.

    Operates directly on the predicted ephemeris (already matched to the
    reduced cube's own frame list -- see match_ephemeris_to_reduced_frames),
    not on any forced-photometry output: measure_stack_centroid_offset
    works from raw cube stamps and returns its own fit's significance, so
    no photometry needs to run before the offset can be measured (see
    asteroid_lightcurves, which now runs forced photometry exactly once,
    at the corrected position this returns, rather than once to find the
    offset and again to apply it).

    Falls back to (nan, nan, 0) if fewer than min_tracks qualify, so
    callers can fall back to a different estimate (e.g.
    asteroid_photometry.identify_known_asteroids's own detected-source
    pooling) rather than trust too few points.

    Also returns a per-track diagnostics DataFrame (every track with at
    least one usable stamp, not just the ones that qualified) with
    designation, offset_x, offset_y,
    sig, flux, and n_stamps, plus mag/mag_err converted from flux via
    zp_ab if a zeropoint is given (mag_err from the standard
    1.0857/sig error-propagation relation -- exact given sig is flux/e_flux
    by construction, no separate uncertainty needed) and a `used` flag
    marking which tracks fed the pooled offset -- so the trust decision is
    inspectable after the fact instead of only the final pooled number."""
    rows = []
    for designation, track in ephemeris.groupby("designation"):
        ox, oy, sig, flux, n_stamps = measure_stack_centroid_offset(
            track, cube, sector, cam, ccd, ccd_x0, ccd_y0, prf_path=prf_path, **stamp_kwargs)
        used = bool(np.isfinite(ox) and np.isfinite(sig) and sig >= sig_threshold)
        if zp_ab is not None and flux is not None and np.isfinite(flux) and flux > 0:
            mag = -2.5 * np.log10(flux) + zp_ab
        else:
            mag = np.nan
        mag_err = 1.0857 / sig if np.isfinite(sig) and sig > 0 else np.nan
        if n_stamps == 0:
            continue
        rows.append(dict(designation=designation, offset_x=ox, offset_y=oy, sig=sig,
                          flux=flux, mag=mag, mag_err=mag_err, n_stamps=n_stamps, used=used))
    # pd.DataFrame([]) has no columns at all (not even 'used') -- a cut where every single
    # predicted track has zero usable stamps (confirmed: a real cut, no tracks ever land
    # in-bounds) would otherwise raise KeyError on the diagnostics["used"] access below
    # instead of just producing zero used tracks, same pitfall already guarded against in
    # forced_aperture_photometry/forced_psf_photometry's own empty-result cases
    diagnostics = pd.DataFrame(rows, columns=["designation", "offset_x", "offset_y", "sig",
                                                "flux", "mag", "mag_err", "n_stamps", "used"])

    used_offsets = diagnostics[diagnostics["used"]]
    if len(used_offsets) < min_tracks:
        return np.nan, np.nan, len(used_offsets), diagnostics
    return (float(used_offsets["offset_x"].median()), float(used_offsets["offset_y"].median()),
            len(used_offsets), diagnostics)


# ---------------------------------------------------------------------------
# Expected apparent magnitude (for reporting alongside measured photometry)
# ---------------------------------------------------------------------------

def measured_apparent_magnitude(flux, zp, sig_threshold=3.0, sig=None):
    """Median apparent magnitude from detections above sig_threshold (if
    sig is given), else from all positive flux -- for reporting alongside
    each track's mag_expected (see asteroid_prediction.expected_apparent_magnitude)."""
    flux = np.asarray(flux, dtype=float)
    if sig is not None:
        flux = flux[np.asarray(sig) > sig_threshold]
    flux = flux[flux > 0]
    if len(flux) == 0:
        return np.nan
    return zp - 2.5 * math.log10(np.median(flux))


# ---------------------------------------------------------------------------
# Cross-matching TESSELLATE's own detections against known asteroids
# ---------------------------------------------------------------------------

MATCH_MAX_DIST_PX = 3.0
MATCH_OFFSET_SEARCH_PX = 5.0
MATCH_MIN_OFFSET_PAIRS = 5


def _pool_cut_offset(detected_df, pred_groups, offset_search_px, min_offset_pairs,
                       detected_id_col, detected_x_col, detected_y_col):
    """Robust per-cut (dx, dy) offset estimate, pooled from every
    (detected, predicted) pair within offset_search_px across the WHOLE
    cut -- not fit per detected object. A single asteroid detection only
    has a handful of noisy points, nowhere near enough to trust a fitted
    offset from; pooling across every object in the cut gives far more
    data for the same underlying systematic (ephemeris/WCS residual is a
    property of the cut/epoch, not of any one object). Falls back to
    zero offset if too few pairs fall within the search radius to trust
    a pooled estimate."""
    all_dx, all_dy = [], []
    for _, det in detected_df.groupby(detected_id_col):
        det = det.sort_values("frame")
        for pred in pred_groups.values():
            merged = det.merge(pred, on="frame", how="inner")
            if len(merged) == 0:
                continue
            dx = merged[detected_x_col].values - merged["x"].values
            dy = merged[detected_y_col].values - merged["y"].values
            close = np.hypot(dx, dy) <= offset_search_px
            all_dx.extend(dx[close])
            all_dy.extend(dy[close])

    if len(all_dx) < min_offset_pairs:
        return 0.0, 0.0
    return float(np.median(all_dx)), float(np.median(all_dy))


def identify_known_asteroids(detected_df, predicted_df, max_dist_px=MATCH_MAX_DIST_PX,
                               offset_search_px=MATCH_OFFSET_SEARCH_PX,
                               min_offset_pairs=MATCH_MIN_OFFSET_PAIRS,
                               detected_id_col="objid",
                               detected_x_col="xcentroid", detected_y_col="ycentroid",
                               offset_x=None, offset_y=None):
    """Proximity cross-match between TESSELLATE's own detections and
    predicted asteroid positions at the same frame, correcting for the
    real systematic offset between predicted and measured position
    (confirmed empirically, up to ~1px even for a validated real object)
    -- without fitting that offset per detected object, since a single
    asteroid detection's few, noisy points aren't reliable enough for
    that.

    offset_x/offset_y let the caller pass in a precomputed cut offset --
    intended for pool_offset_from_stacks' shift-and-stack image centroid
    measurement, which is far more precise than anything derivable from
    TESSELLATE's own noisy per-frame detected centroids (the two tracks it
    was validated on, 45124 and 77999, both had SNR too high for a
    detected-centroid fit to compete). If left as None (the default), the
    offset falls back to _pool_cut_offset's raw-detected-position pooling
    across the whole cut instead.

    For each detected object, checks every frame it shares with each
    candidate predicted asteroid (after shifting the prediction by the
    cut offset) and keeps the single closest approach; a detected
    object counts as matched if that closest approach is within
    max_dist_px, and the closest match overall wins if more than one
    asteroid clears the threshold.

    Returns one row per detected object: the best-matching designation
    (None if nothing cleared max_dist_px), the matching distance, the
    frame it occurred at, and the cut_offset_x/y actually applied
    (same for every row -- useful for sanity-checking the correction).
    Join back onto detected_sources/detected_events/detected_objects by
    detected_id_col."""
    pred_groups = {name: g.sort_values("frame")[["frame", "x", "y"]]
                   for name, g in predicted_df.groupby("designation")}

    if offset_x is None or offset_y is None or not (np.isfinite(offset_x) and np.isfinite(offset_y)):
        offset_x, offset_y = _pool_cut_offset(detected_df, pred_groups, offset_search_px, min_offset_pairs,
                                                detected_id_col, detected_x_col, detected_y_col)

    rows = []
    for obj_id, det in detected_df.groupby(detected_id_col):
        det = det.sort_values("frame")
        best = None
        for designation, pred in pred_groups.items():
            merged = det.merge(pred, on="frame", how="inner")
            if len(merged) == 0:
                continue

            dist = np.hypot(merged[detected_x_col].values - (merged["x"].values + offset_x),
                             merged[detected_y_col].values - (merged["y"].values + offset_y))
            i = int(np.argmin(dist))
            min_dist = float(dist[i])
            if min_dist > max_dist_px:
                continue

            candidate = dict(designation=designation, dist_px=min_dist,
                              frame=int(merged["frame"].values[i]))
            if best is None or candidate["dist_px"] < best["dist_px"]:
                best = candidate

        row = {detected_id_col: obj_id}
        if best is not None:
            row.update(best)
            row["matched_known_asteroid"] = True
        else:
            row.update(dict(designation=None, dist_px=np.nan, frame=None))
            row["matched_known_asteroid"] = False
        row["cut_offset_x"] = offset_x
        row["cut_offset_y"] = offset_y
        rows.append(row)

    return pd.DataFrame(rows)

"""
Predict every catalogued minor planet that crosses a given TESS footprint
at any point during its observing window, using JPL's MPCORB catalogue for
orbital elements and ASSIST (REBOUND's ephemeris-quality N-body integrator,
JPL DE440 + 16 massive asteroid perturbers) for perturbed, light-time
corrected positions from TESS's own real orbital position -- not Earth's
centre, which for TESS (up to ~373,000 km from Earth) is a real, arcsecond-
to-arcminute-scale error for disambiguating and precisely locating objects.

Runs as its own stage in the Tessellate pipeline, per cut, after
make_cuts() and before reduce() -- it only needs the cut's sky footprint
and the sector's observing window, not any reduced pixel data.

Two-stage filter keeps this tractable against MPCORB's ~1.5 million
entries:

  1. Ecliptic-latitude reachability (query.ecliptic_reachable_mask): over a
     full orbit, true anomaly sweeps the entire 0-360 deg range, so the
     maximum ecliptic latitude an object can ever reach is just its
     inclination -- independent of eccentricity or argument of perihelion.
     A single vectorised comparison across the whole catalogue, no
     propagation at all. TESS often points at high ecliptic latitude
     (avoiding zodiacal light), so this alone eliminates most of the
     catalogue for many fields.

  2. Coarse unperturbed 2-body position check (query.coarse_position_mask)
     at several sample epochs across the observing window, for whatever
     survives stage 1. Unperturbed propagation error is negligible over a
     ~27-day sector window (unlike the multi-year archival case this
     module's approach was originally developed against -- see
     precise_ephemeris's own perturbed integration for that regime), so
     this is a reliable coarse filter, not just an approximation of one.

Survivors of both filters get the full ASSIST-perturbed, light-time
corrected integration (precise_ephemeris) across every frame of the
observing window, restricted to frames where the object is actually within
the footprint. Forced photometry is done at the predicted position on every
such frame, regardless of whether the object is independently bright enough
to trigger the pipeline's own transient search -- this gives real
TESS-measured brightness, not just the MPC's catalogue H-magnitude.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u


TESS_HORIZONS_ID = "@-95"  # TESS's spacecraft ID in JPL Horizons/SPICE
AU_KM = const.au.to(u.km).value
LIGHT_TIME_AU_DAYS = (const.au / const.c).to(u.day).value  # light travel time for 1 AU


# ---------------------------------------------------------------------------
# Data loading (cached at module level -- MPCORB is ~316MB/1.5M rows, ASSIST's
# ephemeris files are the better part of 1GB; both are expensive to reload)
# ---------------------------------------------------------------------------

_mpcorb_cache = None
_assist_ephem_cache = None
_spice_furnished = set()


def default_data_dir():
    return os.environ.get("TESS_ORBITS_DATA_DIR", "/Users/rridden/Documents/work/data/orbits")


def load_mpcorb(data_dir=None):
    """Load (and cache) the MPCORB dataframe with derived ecliptic-
    reachability info attached."""
    global _mpcorb_cache
    if _mpcorb_cache is not None:
        return _mpcorb_cache

    from skyfield.data import mpc

    data_dir = data_dir or default_data_dir()
    path = f"{data_dir}/mpc/MPCORB.DAT"
    with open(path, "rb") as f:
        df = mpc.load_mpcorb_dataframe(f)

    # skyfield's parser leaves every column as raw strings (object dtype),
    # including the orbital elements themselves -- cast everything the
    # Kepler solve and brightness filter actually need
    numeric_cols = ["semimajor_axis_au", "eccentricity", "inclination_degrees",
                     "longitude_of_ascending_node_degrees", "argument_of_perihelion_degrees",
                     "mean_anomaly_degrees", "mean_daily_motion_degrees", "magnitude_H", "magnitude_G"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # magnitude_H/G are allowed to be NaN (H handled explicitly in
    # brightness_reachable_mask; G defaults to the standard 0.15 assumption
    # when absent -- see expected_apparent_magnitude); everything else is
    # required for the orbit itself
    required_cols = [c for c in numeric_cols if c not in ("magnitude_H", "magnitude_G")]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    _mpcorb_cache = df
    return df


def load_assist_ephem(data_dir=None):
    global _assist_ephem_cache
    if _assist_ephem_cache is not None:
        return _assist_ephem_cache
    import assist

    data_dir = data_dir or default_data_dir()
    _assist_ephem_cache = assist.Ephem(
        f"{data_dir}/assist/linux_p1550p2650.440", f"{data_dir}/assist/sb441-n16.bsp",
    )
    return _assist_ephem_cache


def furnish_spice_generic(data_dir=None):
    import spiceypy as sp
    data_dir = data_dir or default_data_dir()
    path = f"{data_dir}/generic/naif0012.tls"
    if path not in _spice_furnished:
        sp.furnsh(path)
        _spice_furnished.add(path)


_kernel_coverage_cache = []  # list of (start_isot, end_isot, path), one entry per known segment


def _local_kernel_coverage(data_dir):
    """Merged (start_isot, end_isot) coverage intervals from every TESS SPK
    kernel segment already cached in data_dir/tess_kernels, read directly
    off disk (spkcov) without furnishing or downloading anything. Used by
    verify_data_available to check offline what MJD range is actually
    available."""
    import spiceypy as sp

    kernel_dir = f"{data_dir}/tess_kernels"
    intervals = []
    if not os.path.isdir(kernel_dir):
        return intervals
    furnish_spice_generic(data_dir)  # et2utc below needs the leapsecond kernel furnished first
    for fname in sorted(os.listdir(kernel_dir)):
        if not fname.endswith(".bsp"):
            continue
        path = f"{kernel_dir}/{fname}"
        try:
            cover = sp.spkcov(path, -95)
            for i in range(sp.wncard(cover)):
                b, e = sp.wnfetd(cover, i)
                intervals.append((sp.et2utc(b, "ISOC", 0), sp.et2utc(e, "ISOC", 0)))
        except Exception:
            continue

    intervals.sort()
    merged = []
    for b, e in intervals:
        if merged and b <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((b, e))
    return merged


def verify_data_available(data_dir=None, frame_mjds=None):
    """Raise a clear RuntimeError up front if any file
    predict_asteroids_for_footprint needs isn't already cached locally --
    MPCORB, the ASSIST/DE440 ephemeris files, the generic leapseconds
    kernel, and TESS SPK coverage for every epoch in frame_mjds.

    SLURM compute nodes (unlike login nodes) have no internet access, so
    a missing file can't just be downloaded on demand mid-job the way
    get_tess_kernel_for_epoch does when allowed to -- it needs to be
    caught here, before the job spends any time on the coarse filters or
    spins up worker processes that would each independently fail (or hang
    on a dead connection) trying to reach MAST. Fetch missing kernels with
    fetch_tess_kernels_for_range on a login node first.
    """
    data_dir = data_dir or default_data_dir()
    missing = []

    if not os.path.exists(f"{data_dir}/mpc/MPCORB.DAT"):
        missing.append(f"MPCORB catalogue: {data_dir}/mpc/MPCORB.DAT")
    for fname in ("linux_p1550p2650.440", "sb441-n16.bsp"):
        if not os.path.exists(f"{data_dir}/assist/{fname}"):
            missing.append(f"ASSIST ephemeris file: {data_dir}/assist/{fname}")
    if not os.path.exists(f"{data_dir}/generic/naif0012.tls"):
        missing.append(f"Leap-second kernel: {data_dir}/generic/naif0012.tls")

    if missing:
        raise RuntimeError(
            "asteroid_prediction: missing required data file(s):\n  " + "\n  ".join(missing) +
            "\nFetch these on a login node with internet access before submitting this "
            "stage -- compute nodes cannot reach the internet to download them on demand."
        )

    if frame_mjds is not None and len(frame_mjds):
        import bisect
        from astropy.time import Time

        coverage = _local_kernel_coverage(data_dir)
        if not coverage:
            raise RuntimeError(
                f"asteroid_prediction: no TESS SPK kernels found in {data_dir}/tess_kernels -- "
                "fetch coverage for this date range on a login node first, e.g. "
                "fetch_tess_kernels_for_range(mjd_min, mjd_max, data_dir)."
            )

        starts = [b for b, _ in coverage]
        t_iso = Time(np.atleast_1d(frame_mjds), format="mjd").isot

        def _covered(t):
            i = bisect.bisect_right(starts, t) - 1
            return i >= 0 and coverage[i][0] <= t <= coverage[i][1]

        uncovered = sorted({t for t in t_iso if not _covered(t)})
        if uncovered:
            raise RuntimeError(
                f"asteroid_prediction: no local TESS SPK kernel covers {len(uncovered)} requested "
                f"epoch(s), e.g. {uncovered[0]} (and {uncovered[-1]} if different) -- "
                f"data_dir={data_dir}/tess_kernels. Fetch coverage for this date range on a login "
                "node first, e.g. fetch_tess_kernels_for_range(mjd_min, mjd_max, data_dir)."
            )


def fetch_tess_kernels_for_range(mjd_min, mjd_max, data_dir=None):
    """Download every TESS SPK kernel segment (from MAST) needed to cover
    [mjd_min, mjd_max], skipping any already present locally. Login-node
    only -- requires internet access that SLURM compute nodes don't have.
    Run this ahead of time (e.g. once per sector, or once for a whole
    extended-mission block) so verify_data_available/predict_asteroids
    find full coverage already cached when the actual job runs."""
    import re
    import requests
    from astropy.time import Time

    data_dir = data_dir or default_data_dir()
    kernel_dir = f"{data_dir}/tess_kernels"
    os.makedirs(kernel_dir, exist_ok=True)

    lo_dt, hi_dt = Time(mjd_min, format="mjd").datetime, Time(mjd_max, format="mjd").datetime
    # buffer a week either side -- segment boundaries don't line up with mjd_min/mjd_max exactly,
    # and the filename's day-of-year marks the *end* of a segment's coverage (see
    # get_tess_kernel_for_epoch), so the segment actually needed for mjd_min can be named
    # several days earlier than mjd_min itself
    lo = f"{lo_dt.year:04d}{(lo_dt.timetuple().tm_yday - 7):03d}" if lo_dt.timetuple().tm_yday > 7 \
        else f"{lo_dt.year - 1:04d}{365:03d}"
    hi = f"{hi_dt.year:04d}{(hi_dt.timetuple().tm_yday + 7):03d}"

    listing_url = "https://archive.stsci.edu/missions/tess/models/"
    resp = requests.get(listing_url, timeout=30)
    candidates = sorted(set(re.findall(r'TESS_EPH_DEF_(\d{7})_(\d+)\.bsp', resp.text)))
    in_range = [(code, ver) for code, ver in candidates if lo <= code <= hi]

    best = {}
    for code, ver in in_range:
        if code not in best or int(ver) > int(best[code]):
            best[code] = ver

    existing = set(os.listdir(kernel_dir))
    to_download = [f"TESS_EPH_DEF_{code}_{ver}.bsp" for code, ver in sorted(best.items())
                   if f"TESS_EPH_DEF_{code}_{ver}.bsp" not in existing]

    print(f"fetch_tess_kernels_for_range: downloading {len(to_download)} segment(s)...", flush=True)
    ok, failed = 0, []
    for fname in to_download:
        try:
            r = requests.get(f"{listing_url}{fname}", timeout=120)
            r.raise_for_status()
            with open(f"{kernel_dir}/{fname}", "wb") as f:
                f.write(r.content)
            ok += 1
        except Exception as ex:
            failed.append((fname, str(ex)))
    print(f"fetch_tess_kernels_for_range: {ok} downloaded, {len(failed)} failed.", flush=True)
    for fname, err in failed:
        print(f"  FAILED {fname}: {err}", flush=True)
    return ok, failed


def get_tess_kernel_for_epoch(mjd, data_dir=None, allow_download=True):
    """Return a locally-cached TESS SPK kernel path covering the given MJD,
    downloading it from MAST on first use. MAST publishes one segment file
    every few days across the whole mission (~2400 total) -- fetching on
    demand and caching avoids bulk-downloading all of them.
    File-naming convention: TESS_EPH_DEF_<year><day-of-year>_<version>.bsp,
    where the numeric part is the *end* of the segment's coverage, not the
    start (confirmed empirically: TESS_EPH_DEF_2021322_21.bsp covers
    2021-11-04 to 2021-11-18, i.e. ends on day 322 = Nov 18).

    Coverage windows are cached in-process (`_kernel_coverage_cache`) and
    each kernel is furnished at most once (`_spice_furnished`) -- this is
    called once per frame from `precise_ephemeris`'s per-object loop, so
    without caching it would re-scan and re-furnish every segment on every
    single frame, which both wastes time and eventually exhausts SPICE's
    fixed 5300-kernel KEEPER limit. Checking a file's coverage via
    `spkcov` does not require furnishing it first.
    """
    import spiceypy as sp
    from astropy.time import Time
    import requests
    import re

    data_dir = data_dir or default_data_dir()
    kernel_dir = f"{data_dir}/tess_kernels"
    os.makedirs(kernel_dir, exist_ok=True)
    furnish_spice_generic(data_dir)

    t = Time(mjd, format="mjd")
    t_iso = t.isot

    def _furnish(path):
        if path not in _spice_furnished:
            sp.furnsh(path)
            _spice_furnished.add(path)

    for b_iso, e_iso, path in _kernel_coverage_cache:
        if b_iso <= t_iso <= e_iso:
            _furnish(path)
            return path

    # scan local files not yet catalogued in the coverage cache
    known_paths = {p for _, _, p in _kernel_coverage_cache}
    for fname in sorted(os.listdir(kernel_dir)):
        if not fname.endswith(".bsp"):
            continue
        path = f"{kernel_dir}/{fname}"
        if path in known_paths:
            continue
        try:
            cover = sp.spkcov(path, -95)
            for i in range(sp.wncard(cover)):
                b, e = sp.wnfetd(cover, i)
                _kernel_coverage_cache.append((sp.et2utc(b, "ISOC", 0), sp.et2utc(e, "ISOC", 0), path))
        except Exception:
            continue

    for b_iso, e_iso, path in _kernel_coverage_cache:
        if b_iso <= t_iso <= e_iso:
            _furnish(path)
            return path

    if not allow_download:
        raise RuntimeError(
            f"asteroid_prediction: no local TESS SPK kernel covers MJD {mjd} ({t_iso}) in "
            f"{kernel_dir}, and allow_download=False (this is the normal setting on SLURM "
            "compute nodes, which have no internet access). Fetch coverage for this date range "
            "on a login node first, e.g. fetch_tess_kernels_for_range(mjd_min, mjd_max, data_dir)."
        )

    # not cached -- find and download the right segment from MAST's listing.
    # The filename's day-of-year is only a coarse marker for where coverage
    # ends -- the true end-of-coverage instant can be hours before that
    # day's end (e.g. a segment named "day 330" can actually stop at
    # 13:00 UTC on day 330), so a same-day match can still fall short of
    # the target epoch. Verify actual coverage after each candidate and
    # advance to the next one if it doesn't reach far enough.
    year = t.datetime.year
    doy = t.datetime.timetuple().tm_yday
    listing_url = "https://archive.stsci.edu/missions/tess/models/"
    resp = requests.get(listing_url, timeout=30)
    candidates = sorted(set(re.findall(r'TESS_EPH_DEF_(\d{7})_(\d+)\.bsp', resp.text)))
    target_code = f"{year:04d}{doy:03d}"
    start_idx = next((i for i, (code, _) in enumerate(candidates) if code >= target_code), None)
    if start_idx is None:
        raise ValueError(f"No TESS SPK kernel found covering MJD {mjd} (year {year} day {doy})")

    for code, version in candidates[start_idx:]:
        fname = f"TESS_EPH_DEF_{code}_{version}.bsp"
        path = f"{kernel_dir}/{fname}"
        if not os.path.exists(path):
            r = requests.get(f"{listing_url}{fname}", timeout=120)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        cover = sp.spkcov(path, -95)
        covers_target = False
        for i in range(sp.wncard(cover)):
            b, e = sp.wnfetd(cover, i)
            b_iso, e_iso = sp.et2utc(b, "ISOC", 0), sp.et2utc(e, "ISOC", 0)
            _kernel_coverage_cache.append((b_iso, e_iso, path))
            if b_iso <= t_iso <= e_iso:
                covers_target = True
        if covers_target:
            _furnish(path)
            return path

    raise ValueError(f"No TESS SPK kernel segment actually covers MJD {mjd} (year {year} day {doy})")


# ---------------------------------------------------------------------------
# Stage 1: ecliptic-latitude reachability
# ---------------------------------------------------------------------------

def ecliptic_reachable_mask(mpcorb_df, epoch_mjd, sample_mjds, target_ecl_lat_deg, margin_deg):
    """Vectorised, propagated -- an object's REAL ecliptic latitude at the actual observation
    epoch, not a static geometric bound.

    The previous version compared the field's ecliptic latitude, as seen from EARTH, against
    only the object's HELIOCENTRIC orbital inclination: `inclination + margin_deg >=
    abs(target_ecl_lat_deg)`. Those are not the same quantity. Earth sits up to 1 au off the
    Sun-object line, and that parallax shifts the apparent ecliptic latitude beyond what the
    heliocentric inclination alone permits -- by asin(1 au / perihelion distance), which is
    negligible for a very distant object but ~11 deg for a Jupiter Trojan and tens of degrees
    for an NEO. A flat 1 deg margin does not begin to cover that.

    Confirmed missing (5254) Ulysses (i=24.19 deg) from a field at ecliptic latitude -30.6 deg:
    24.19+1.0 < 30.6 by the old bound, so it was dropped at this first stage, before
    propagation or brightness were ever checked, even though its real position that epoch sits
    inside the field. The object was independently confirmed present in TESS's own archive by a
    separate, non-TESSELLATE detection (SkyBoT cone search against the raw images).

    A parallax-bounded worst case fixes the miss but is far too loose to use as a filter --
    measured against the real MPCORB catalogue, it inflates survivors at this field's latitude
    from 14,184 to 1,251,467 (99.7th percentile: 213,186), which would blow out every
    downstream stage's cost by 1-2 orders of magnitude for any field off the ecliptic plane.

    So instead of bounding what the orbit could EVER reach, this propagates each object's
    actual position to sample_mjds (the same coarse-sampling epochs stage 2 already uses,
    reusing _vectorized_kepler_unit_vectors -- the identical machinery stage 2 calls, just run
    over the full catalogue instead of stage 1b's survivors, ~4-7s for 1.5M rows) and tests the
    real ecliptic latitude at each sample against the field's, keeping a row if ANY sample
    comes within margin_deg. margin_deg should be sized the same way the caller sizes stage 2's
    (footprint radius + motion budget over the sample spacing), not the whole-window motion
    budget, since a handful of samples across the window already bound how far a real position
    can drift between them.
    """
    x, y, z = _vectorized_kepler_unit_vectors(mpcorb_df, epoch_mjd, np.asarray(sample_mjds))
    lat_deg = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))          # (n_objects, n_samples)
    return (np.abs(lat_deg - target_ecl_lat_deg) <= margin_deg).any(axis=1)


# ---------------------------------------------------------------------------
# Stage 1b: best-case brightness reachability
# ---------------------------------------------------------------------------

def brightness_reachable_mask(mpcorb_df, faint_limit_mag=20.8):
    """Vectorised, no propagation: most of MPCORB's ~1.5 million entries
    are small, distant objects that could never be bright enough for TESS
    regardless of position -- this is a far more powerful cut than the
    ecliptic filter alone (which only removed ~17% of the catalogue for a
    real test field). Estimates each object's best-case apparent
    magnitude from its absolute magnitude H and perihelion distance q
    (closest possible heliocentric distance) paired with the closest
    plausible geocentric distance (opposition-like, q - 1 AU, floored),
    ignoring the phase-angle brightening correction -- deliberately
    optimistic, so this only ever discards objects that are provably too
    faint under ANY geometry, never a real risk of dropping something
    genuinely detectable.

    Default is TESS's own limiting magnitude, ~20.8 in I-band. MPCORB's H
    is defined in a roughly V-band-like system, and asteroids are
    typically somewhat redder than solar colour (S-/C-type V-I of order a
    few tenths of a mag), so they generally read *brighter* in I than a
    naive V-band comparison suggests -- comparing the raw H-based estimate
    against the I-band limit is if anything conservative (safe), not the
    other way round; no per-object colour correction is applied since
    MPCORB doesn't reliably carry taxonomic type for most entries. If the
    stage-1b survivor count is still too large to be practical for stage
    2's propagation, tighten this to 20.0 rather than push it looser."""
    a = mpcorb_df["semimajor_axis_au"].values
    e = mpcorb_df["eccentricity"].values
    H = pd.to_numeric(mpcorb_df["magnitude_H"], errors="coerce").values
    q = a * (1 - e)
    d_geo_best = np.maximum(q - 1.0, 0.1)
    best_case_mag = H + 5 * np.log10(q * d_geo_best)
    # unknown H (rare) can't be ruled out -- keep rather than risk a false negative
    return np.isnan(H) | (best_case_mag <= faint_limit_mag)


# ---------------------------------------------------------------------------
# Epoch decoding (vectorised)
# ---------------------------------------------------------------------------

def _mpc_packed_epoch_to_mjd(epoch_packed):
    """Vectorised decode of MPC's 5-character packed epoch date
    (e.g. 'K2669' -> 2026-06-09) into MJD. A handful of MPCORB rows have
    non-standard epoch encodings (comet-heritage or malformed entries) --
    those decode to NaN and get dropped rather than raising."""
    from skyfield.data.mpc import julian_day

    s = pd.Series(epoch_packed)
    valid = s.str.match(r"^[A-Za-z0-9][0-9]{2}[A-Za-z0-9]{2}$", na=False)

    def n_char(c):
        return np.where(np.char.isdigit(c), np.array(list(c)).view(np.int32) - 48,
                         np.array(list(c)).view(np.int32) - 55)

    mjd = np.full(len(s), np.nan)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return mjd

    chars = s.iloc[idx].values
    c0 = np.array([c[0] for c in chars])
    c1_3 = np.array([c[1:3] for c in chars])
    c3 = np.array([c[3] for c in chars])
    c4 = np.array([c[4] for c in chars])

    def n(c_arr):
        return np.array([ord(c) - (48 if c.isdigit() else 55) for c in c_arr])

    year = 100 * n(c0) + c1_3.astype(int)
    month = n(c3)
    day = n(c4)

    jd = np.array([julian_day(y, m, d) for y, m, d in zip(year, month, day)]) - 0.5
    mjd[idx] = jd - 2400000.5
    return mjd


# ---------------------------------------------------------------------------
# Stage 2: coarse unperturbed 2-body position check (fully vectorised)
# ---------------------------------------------------------------------------

def _earth_heliocentric_ecliptic_au(sample_mjds):
    """Earth's heliocentric position in the barycentric-true-ecliptic
    frame (AU), one vector per sample epoch. Needed because a main-belt
    asteroid is typically only ~1-2 AU from Earth -- comparable to its
    ~2-3 AU heliocentric distance -- so the parallax between "direction
    from the Sun" and "direction from Earth" is tens of degrees, not a
    negligible correction (confirmed empirically: ignoring it produced a
    22 deg error for a real object even with zero propagation time)."""
    from astropy.coordinates import get_body_barycentric, SkyCoord, BarycentricTrueEcliptic
    from astropy.time import Time
    import astropy.units as u

    t = Time(np.asarray(sample_mjds), format="mjd")
    earth_bary = get_body_barycentric("earth", t)
    sun_bary = get_body_barycentric("sun", t)
    helio = earth_bary - sun_bary
    c = SkyCoord(x=helio.x, y=helio.y, z=helio.z, frame="icrs", representation_type="cartesian")
    c_ecl = c.transform_to(BarycentricTrueEcliptic())
    xyz = c_ecl.cartesian.xyz.to(u.au).value  # (3, T)
    return xyz


def _vectorized_kepler_unit_vectors(mpcorb_df, epoch_mjd, sample_mjds, earth_helio_au=None):
    """Geocentric (topocentric-approximate) ecliptic unit-vectors for
    every object in mpcorb_df at every sample time, computed all at once
    via a vectorised Newton-Raphson Kepler solve -- not a per-object loop.
    Subtracts Earth's heliocentric position (see
    _earth_heliocentric_ecliptic_au) before normalising, since that
    offset is not negligible for typical main-belt distances. Still
    ignores TESS's own small (<400,000 km << 1 AU) offset from Earth's
    centre -- that part genuinely is negligible for this coarse filter.
    Returns arrays of shape (N_objects, N_times)."""
    a = mpcorb_df["semimajor_axis_au"].values[:, None]
    e = mpcorb_df["eccentricity"].values[:, None]
    inc = np.radians(mpcorb_df["inclination_degrees"].values)[:, None]
    node = np.radians(mpcorb_df["longitude_of_ascending_node_degrees"].values)[:, None]
    peri = np.radians(mpcorb_df["argument_of_perihelion_degrees"].values)[:, None]
    M0 = np.radians(mpcorb_df["mean_anomaly_degrees"].values)[:, None]
    n = np.radians(mpcorb_df["mean_daily_motion_degrees"].values)[:, None]

    dt = np.asarray(sample_mjds)[None, :] - epoch_mjd[:, None]
    M = np.mod(M0 + n * dt, 2 * np.pi)

    E = M.copy()
    for _ in range(10):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    x_orb, y_orb = r * np.cos(nu), r * np.sin(nu)

    cos_O, sin_O = np.cos(node), np.sin(node)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_w, sin_w = np.cos(peri), np.sin(peri)

    x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb
    y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    if earth_helio_au is None:
        earth_helio_au = _earth_heliocentric_ecliptic_au(sample_mjds)
    x = x - earth_helio_au[0][None, :]
    y = y - earth_helio_au[1][None, :]
    z = z - earth_helio_au[2][None, :]

    norm = np.sqrt(x**2 + y**2 + z**2)
    return x / norm, y / norm, z / norm


# ---------------------------------------------------------------------------
# Per-sector MPCORB snapshot: re-epoch the whole catalogue to a sector's own
# midpoint via a real perturbed integration, so the coarse stages above never
# see a multi-year gap between MPCORB's catalogue epoch and the observing
# window.
#
# The module docstring above says unperturbed propagation is "negligible over
# a ~27-day sector window" -- true, but it assumes epoch_mjd already sits near
# that window. MPCORB.DAT is a single, continuously-refreshed file with no
# per-sector archive; reprocessing an old sector with today's file can leave
# a multi-year gap. Confirmed on (1) Ceres: MPCORB's epoch was 2112 days
# (more than one Ceres year) from a real Sector 29 window, and 2-body
# propagation across that gap put it 15 deg from its true (perturbed)
# position -- a difference of that size routinely flips a coarse-stage
# in/out decision.
#
# Building a full ASSIST integration is exactly what precise_ephemeris
# already does, but per-survivor; this applies it to the WHOLE catalogue
# once per sector; splitting off close-perihelion objects only for the
# fast timestep and cost analysis they force, everything else is one, we
# get the same accuracy without the cost.
# ---------------------------------------------------------------------------

NEO_PERIHELION_AU = 1.3  # standard NEO definition; also where REBOUND's adaptive
                          # timestep starts being forced down by close passages


def _vectorized_state_at_own_epoch(mpcorb_df):
    """Heliocentric ecliptic (position_au, velocity_au_per_day) for every row, AT ITS OWN
    MPCORB reference epoch -- exact 2-body, not an approximation, because dt=0 there: mean
    anomaly at an object's own epoch is exactly its catalogued mean_anomaly_degrees, with no
    Kepler-equation-vs-perturbation drift to accumulate. The one iterative step still needed
    (mean anomaly -> eccentric anomaly) is the same Newton solve _vectorized_kepler_unit_vectors
    already uses; this returns full 3-D state (both position AND velocity), not just a
    direction, so the result can seed a REBOUND simulation.

    mu (GM_sun) is not a separate constant: MPCORB's mean_daily_motion_degrees already encodes
    it per Kepler's third law (n^2 = mu/a^3), so mu = n^2 a^3 recovers exactly the GM_sun value
    consistent with each object's own published elements -- no unit mismatch is possible.
    """
    a = mpcorb_df["semimajor_axis_au"].values
    e = mpcorb_df["eccentricity"].values
    inc = np.radians(mpcorb_df["inclination_degrees"].values)
    node = np.radians(mpcorb_df["longitude_of_ascending_node_degrees"].values)
    peri = np.radians(mpcorb_df["argument_of_perihelion_degrees"].values)
    M = np.radians(mpcorb_df["mean_anomaly_degrees"].values)
    n = np.radians(mpcorb_df["mean_daily_motion_degrees"].values)
    mu = n ** 2 * a ** 3

    E = M.copy()
    for _ in range(10):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    cos_E, sin_E = np.cos(E), np.sin(E)
    r = a * (1 - e * cos_E)
    x_orb = a * (cos_E - e)
    y_orb = a * np.sqrt(1 - e ** 2) * sin_E
    # standard vis-viva-derived orbital-plane velocity (rad/day-consistent since n is rad/day)
    vx_orb = -a * n * sin_E / (1 - e * cos_E)
    vy_orb = a * n * np.sqrt(1 - e ** 2) * cos_E / (1 - e * cos_E)

    cos_O, sin_O = np.cos(node), np.sin(node)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_w, sin_w = np.cos(peri), np.sin(peri)
    R11 = cos_O * cos_w - sin_O * sin_w * cos_i
    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    R21 = sin_O * cos_w + cos_O * sin_w * cos_i
    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    R31 = sin_w * sin_i
    R32 = cos_w * sin_i

    pos = np.stack([R11 * x_orb + R12 * y_orb,
                    R21 * x_orb + R22 * y_orb,
                    R31 * x_orb + R32 * y_orb], axis=-1)
    vel = np.stack([R11 * vx_orb + R12 * vy_orb,
                    R21 * vx_orb + R22 * vy_orb,
                    R31 * vx_orb + R32 * vy_orb], axis=-1)
    return pos, vel


def _state_to_elements(pos_au, vel_au_per_day, mu):
    """Inverse of _vectorized_state_at_own_epoch: classical osculating elements from a
    heliocentric Cartesian state, vectorised, closed-form (no iteration -- unlike mean-to-
    eccentric anomaly, Cartesian-to-elements needs no Kepler-equation solve). Standard
    astrodynamics formulas (e.g. Vallado); mu must be the SAME per-object value used to build
    the state (n^2 a^3 from the ORIGINAL elements), since that is what a REBOUND/ASSIST
    integration implicitly conserves for an unperturbed two-body problem, and using the Sun's
    true GM here instead would reintroduce a small but avoidable inconsistency.

    Returns a dict of the columns load_mpcorb's consumers expect, EXCLUDING epoch (the caller
    sets that to the target epoch this state was integrated to).
    """
    r = np.linalg.norm(pos_au, axis=-1)
    v2 = np.sum(vel_au_per_day ** 2, axis=-1)
    h_vec = np.cross(pos_au, vel_au_per_day)
    h = np.linalg.norm(h_vec, axis=-1)
    node_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    node_mag = np.linalg.norm(node_vec, axis=-1)

    e_vec = (np.cross(vel_au_per_day, h_vec) / mu[..., None]
             - pos_au / r[..., None])
    e = np.linalg.norm(e_vec, axis=-1)

    energy = v2 / 2 - mu / r
    a = -mu / (2 * energy)

    inc = np.arccos(np.clip(h_vec[..., 2] / h, -1, 1))

    node = np.arctan2(node_vec[..., 1], node_vec[..., 0])
    node = np.where(node_mag > 1e-12, np.mod(node, 2 * np.pi), 0.0)

    cos_peri = np.clip(np.sum(node_vec * e_vec, axis=-1) / np.maximum(node_mag * e, 1e-30), -1, 1)
    peri = np.arccos(cos_peri)
    peri = np.where(e_vec[..., 2] < 0, 2 * np.pi - peri, peri)

    cos_nu = np.clip(np.sum(e_vec * pos_au, axis=-1) / np.maximum(e * r, 1e-30), -1, 1)
    nu = np.arccos(cos_nu)
    rdotv = np.sum(pos_au * vel_au_per_day, axis=-1)
    nu = np.where(rdotv < 0, 2 * np.pi - nu, nu)

    E = 2 * np.arctan2(np.sqrt(np.clip(1 - e, 0, None)) * np.sin(nu / 2),
                       np.sqrt(np.clip(1 + e, 0, None)) * np.cos(nu / 2))
    M = np.mod(E - e * np.sin(E), 2 * np.pi)
    n_rad_day = np.sqrt(mu / a ** 3)

    return dict(semimajor_axis_au=a, eccentricity=e,
               inclination_degrees=np.degrees(inc),
               longitude_of_ascending_node_degrees=np.degrees(node),
               argument_of_perihelion_degrees=np.degrees(peri),
               mean_anomaly_degrees=np.degrees(M),
               mean_daily_motion_degrees=np.degrees(n_rad_day))


def build_sector_mpcorb_snapshot(target_mjd, data_dir=None, out_path=None,
                                 shard_index=0, n_shards=1, neo_perihelion_au=NEO_PERIHELION_AU):
    """Re-epoch a shard of MPCORB to target_mjd (a sector's midpoint) via a real ASSIST
    integration, and save it in load_mpcorb's own column format so it is a drop-in replacement
    -- ecliptic_reachable_mask, brightness_reachable_mask and coarse_position_mask need no
    changes, only the epoch_mjd they are handed changes.

    Sharding is over the BULK (q >= neo_perihelion_au) population only, split across n_shards
    equal slices by row order -- safe because REBOUND's adaptive step here is set by the
    slowest-varying, near-circular bulk of the catalogue, not by row order, so any slice is
    representative. The close-perihelion minority (q < neo_perihelion_au, ~2.7% of MPCORB,
    dominated by NEAs) is excluded from every bulk shard and handled once, on shard_index==0,
    as its own separate integration: mixing it into the bulk sweep would force every bulk
    particle down to the NEA-forced timestep, for no benefit (see module history/benchmarks --
    a mixed 5,000-particle sweep cost 24.4 ms/particle where a bulk-only sweep of the same size
    cost 7.8 ms/particle, purely from a handful of small-perihelion outliers).

    The MPCORB minority not on the file's dominant shared epoch (~0.7%, up to a ~5-year-old
    epoch in the current file) is further excluded from both and grouped by its own epoch
    value: each distinct epoch gets its own small integration from THAT epoch to target_mjd,
    since REBOUND requires one shared sim.t for every particle added to a simulation. This is
    a bounded, small Python-level loop (measured: 428 distinct epochs, none of them large),
    not a per-object loop over the catalogue.

    Writes a parquet file of shard results; the caller (or a merge step) concatenates shards.
    """
    data_dir = data_dir or default_data_dir()
    mpcorb = load_mpcorb(data_dir)
    epoch_mjd = _mpc_packed_epoch_to_mjd(mpcorb["epoch_packed"].values)
    q = mpcorb["semimajor_axis_au"].values * (1 - mpcorb["eccentricity"].values)

    common_epoch = pd.Series(epoch_mjd).mode().iloc[0]
    on_common = np.isclose(epoch_mjd, common_epoch)
    is_bulk = on_common & (q >= neo_perihelion_au)
    is_neo = on_common & (q < neo_perihelion_au)
    is_minority = ~on_common

    ephem = load_assist_ephem(data_dir)
    jd_ref = ephem.jd_ref
    t_target_days = (target_mjd + 2400000.5) - jd_ref

    def integrate_group(sub_df, start_epoch_mjd):
        """One REBOUND+ASSIST sweep: everyone in sub_df starts at start_epoch_mjd, all
        integrated together once to t_target_days. Returns sub_df's designation_packed,
        magnitude_H, magnitude_G alongside the re-epoched elements."""
        import assist
        import rebound

        if len(sub_df) == 0:
            return pd.DataFrame(columns=["designation_packed", "magnitude_H", "magnitude_G",
                                         "semimajor_axis_au", "eccentricity",
                                         "inclination_degrees",
                                         "longitude_of_ascending_node_degrees",
                                         "argument_of_perihelion_degrees",
                                         "mean_anomaly_degrees", "mean_daily_motion_degrees",
                                         "epoch_mjd"])
        pos, vel = _vectorized_state_at_own_epoch(sub_df)
        start_days = (start_epoch_mjd + 2400000.5) - jd_ref
        sun = ephem.get_particle("Sun", start_days)
        sun_pos = np.array([sun.x, sun.y, sun.z])
        sun_vel = np.array([sun.vx, sun.vy, sun.vz])

        sim = rebound.Simulation()
        assist.Extras(sim, ephem)
        sim.t = start_days
        bary_pos = pos + sun_pos
        bary_vel = vel + sun_vel
        for i in range(len(sub_df)):
            sim.add(x=bary_pos[i, 0], y=bary_pos[i, 1], z=bary_pos[i, 2],
                   vx=bary_vel[i, 0], vy=bary_vel[i, 1], vz=bary_vel[i, 2])
        sim.integrate(t_target_days)

        end_pos = np.array([[p.x, p.y, p.z] for p in sim.particles])
        end_vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
        sun_end = ephem.get_particle("Sun", t_target_days)
        helio_pos = end_pos - np.array([sun_end.x, sun_end.y, sun_end.z])
        helio_vel = end_vel - np.array([sun_end.vx, sun_end.vy, sun_end.vz])

        n_orig = np.radians(sub_df["mean_daily_motion_degrees"].values)
        a_orig = sub_df["semimajor_axis_au"].values
        mu = n_orig ** 2 * a_orig ** 3          # each object's OWN implied GM_sun, preserved
        elements = _state_to_elements(helio_pos, helio_vel, mu)

        out = pd.DataFrame(elements)
        out["designation_packed"] = sub_df["designation_packed"].values
        out["magnitude_H"] = sub_df["magnitude_H"].values
        out["magnitude_G"] = sub_df["magnitude_G"].values
        out["epoch_mjd"] = target_mjd
        return out

    results = []
    if shard_index == 0:
        # the small groups are cheap and only need doing once, regardless of how many
        # shards the bulk population is split across
        results.append(integrate_group(mpcorb[is_neo].reset_index(drop=True), common_epoch))
        minority = mpcorb[is_minority]
        minority_epochs = epoch_mjd[is_minority]
        for ep in np.unique(minority_epochs):
            grp = minority[np.isclose(minority_epochs, ep)].reset_index(drop=True)
            results.append(integrate_group(grp, ep))

    bulk_idx = np.where(is_bulk)[0]
    my_shard = np.array_split(bulk_idx, n_shards)[shard_index]
    results.append(integrate_group(mpcorb.iloc[my_shard].reset_index(drop=True), common_epoch))

    out_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if out_path:
        out_df.to_parquet(out_path, index=False)
    return out_df


def load_mpcorb_snapshot(path):
    """Load a per-sector re-epoched snapshot from build_sector_mpcorb_snapshot (merged across
    shards). Same column set load_mpcorb produces, so every downstream stage (ecliptic/
    brightness/coarse) works unchanged; epoch_mjd is a plain column here rather than needing
    _mpc_packed_epoch_to_mjd, since the snapshot was written with a real epoch already."""
    df = pd.read_parquet(path)
    required = ["semimajor_axis_au", "eccentricity", "inclination_degrees",
               "longitude_of_ascending_node_degrees", "argument_of_perihelion_degrees",
               "mean_anomaly_degrees", "mean_daily_motion_degrees", "epoch_mjd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"snapshot at {path} is missing columns: {missing}")
    return df


def coarse_position_mask(mpcorb_df, ra_center_deg, dec_center_deg, radius_deg, sample_mjds,
                          return_hit_samples=False):
    """Fully vectorised coarse 2-body position check across every object
    in mpcorb_df at once (see _vectorized_kepler_unit_vectors) -- keeps
    anything within radius_deg of the field centre at any sample epoch.
    Unperturbed 2-body propagation error grows with time-from-epoch, so
    this is only reliable when propagated over a short baseline; the
    margin passed in by the caller should already account for the gap
    between the catalogue epoch and the observing window (see
    predict_asteroids_for_footprint), not just the window's own span.
    Compares in ecliptic coordinates (matching the Kepler solve's native
    frame) rather than converting every object to equatorial RA/Dec.

    return_hit_samples=True also returns, per surviving row (keyed by its
    index into the input mpcorb_df):
      - hit_samples_by_row: the sample_mjds where that object was actually
        near the footprint
      - rate_by_row: that object's own observed angular rate (deg/day,
        the largest great-circle step seen between any two consecutive
        coarse samples). A fixed pad derived from the catalogue-wide
        max_motion_deg_per_day assumption isn't safe for real close/fast
        near-Earth objects, which can exceed it by several times (e.g. a
        NEO at ~0.05 AU can move multiple deg/day, well past the ~1.5
        deg/day the coarse margin assumes for the general population) --
        the caller uses this per-object rate to size how much to pad
        around each hit sample, so fast movers get a wider window instead
        of silently losing frames a full, unrestricted scan would have
        caught (see predict_asteroids_for_footprint)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    epoch_mjd = _mpc_packed_epoch_to_mjd(mpcorb_df["epoch_packed"].values)
    valid = ~np.isnan(epoch_mjd)

    earth_helio_au = _earth_heliocentric_ecliptic_au(sample_mjds)
    x, y, z = _vectorized_kepler_unit_vectors(mpcorb_df[valid].reset_index(drop=True),
                                                 epoch_mjd[valid], sample_mjds, earth_helio_au)

    target_ecl = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg).barycentrictrueecliptic
    tx = math.cos(target_ecl.lat.rad) * math.cos(target_ecl.lon.rad)
    ty = math.cos(target_ecl.lat.rad) * math.sin(target_ecl.lon.rad)
    tz = math.sin(target_ecl.lat.rad)
    cos_radius = math.cos(math.radians(radius_deg))

    dot = x * tx + y * ty + z * tz  # (N, T)
    hit_matrix = dot >= cos_radius  # (N, T), per-object per-sample -- NOT collapsed
    hit_valid = hit_matrix.any(axis=1)

    keep = np.zeros(len(mpcorb_df), dtype=bool)
    keep[np.where(valid)[0][hit_valid]] = True

    if not return_hit_samples:
        return keep

    valid_idx = np.where(valid)[0]
    # angular separation between EVERY consecutive pair of samples (great-circle, via the
    # unit vectors already computed above), regardless of hit/miss -- a fast object can
    # legitimately be a miss at one sample and a hit at the next, and the rate estimate
    # needs the true motion between those two, not just motion between hit samples
    dt = np.diff(sample_mjds)
    cons_dot = np.clip(x[:, :-1] * x[:, 1:] + y[:, :-1] * y[:, 1:] + z[:, :-1] * z[:, 1:], -1.0, 1.0)
    ang_sep_deg = np.degrees(np.arccos(cons_dot))  # (N, T-1)
    rate_deg_per_day = ang_sep_deg / dt[None, :] if len(dt) else np.zeros((len(x), 0))

    hit_idx = np.where(hit_valid)[0]
    hit_samples_by_row = {int(valid_idx[i]): sample_mjds[hit_matrix[i]] for i in hit_idx}
    rate_by_row = {int(valid_idx[i]): float(np.nanmax(rate_deg_per_day[i])) if rate_deg_per_day.shape[1] else 0.0
                   for i in hit_idx}
    return keep, hit_samples_by_row, rate_by_row


# ---------------------------------------------------------------------------
# Precise per-frame ephemeris (validated to ~0.04 arcsec against JPL
# Horizons for a real archival case -- see FRB20211113B / asteroid Aurochs)
# ---------------------------------------------------------------------------

def _state_vector_at_epoch(row, data_dir):
    """Heliocentric state at an object's own reference epoch, from either source of elements:

    - raw MPCORB rows carry epoch_packed (MPC's packed date string) -- unchanged skyfield path.
    - rows from a per-sector snapshot (build_sector_mpcorb_snapshot) carry a plain epoch_mjd
      float instead, since they were already re-epoched to a real date, not an MPC packed
      string. Uses the same vectorised 2-body solve the snapshot builder itself uses
      (_vectorized_state_at_own_epoch), on a length-1 batch -- exact at dt=0 either way, so
      this is not a lower-accuracy fallback, just a different element source."""
    epoch_packed = getattr(row, "epoch_packed", None)
    if epoch_packed is not None and not (isinstance(epoch_packed, float) and math.isnan(epoch_packed)):
        from skyfield.data import mpc
        from skyfield.api import load
        from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2
        from types import SimpleNamespace

        ts = load.timescale()
        ns_row = SimpleNamespace(
            semimajor_axis_au=float(row.semimajor_axis_au), eccentricity=float(row.eccentricity),
            inclination_degrees=float(row.inclination_degrees),
            longitude_of_ascending_node_degrees=float(row.longitude_of_ascending_node_degrees),
            argument_of_perihelion_degrees=float(row.argument_of_perihelion_degrees),
            mean_anomaly_degrees=float(row.mean_anomaly_degrees),
            epoch_packed=epoch_packed, designation=row.designation,
        )
        orbit = mpc.mpcorb_orbit(ns_row, ts, GM_SUN_Pitjeva_2005_km3_s2)
        t_ref = orbit.epoch
        return t_ref, orbit.at(t_ref).position.au, orbit.at(t_ref).velocity.au_per_d

    from skyfield.api import load
    epoch_mjd = float(row.epoch_mjd)
    row_df = pd.DataFrame([dict(
        semimajor_axis_au=row.semimajor_axis_au, eccentricity=row.eccentricity,
        inclination_degrees=row.inclination_degrees,
        longitude_of_ascending_node_degrees=row.longitude_of_ascending_node_degrees,
        argument_of_perihelion_degrees=row.argument_of_perihelion_degrees,
        mean_anomaly_degrees=row.mean_anomaly_degrees,
        mean_daily_motion_degrees=row.mean_daily_motion_degrees)])
    pos, vel = _vectorized_state_at_own_epoch(row_df)
    t_ref = load.timescale().tdb(jd=epoch_mjd + 2400000.5)
    return t_ref, pos[0], vel[0]


DEFAULT_SLOPE_G = 0.15  # MPC's standard assumption when an object's own G is unmeasured


def expected_apparent_magnitude(H, G, r_helio_au, delta_au, phase_angle_deg):
    """Standard IAU H-G system apparent magnitude (approximately V-band,
    the same system MPCORB's magnitude_H is defined in): brightness falls
    off with the square of both the Sun-object and object-observer
    distances, corrected for the phase angle (Sun-object-observer angle)
    via the two-term HG phase function -- a full-phase (alpha=0) object
    is brighter than the inverse-square law alone predicts, since more of
    its illuminated hemisphere faces the observer. Vectorised over arrays;
    G may be a scalar (falls back to DEFAULT_SLOPE_G if NaN/None) or an
    array (NaNs replaced with the default element-wise).
    """
    r_helio_au = np.asarray(r_helio_au, dtype=float)
    delta_au = np.asarray(delta_au, dtype=float)
    alpha = np.radians(np.asarray(phase_angle_deg, dtype=float))

    if G is None:
        G_use = DEFAULT_SLOPE_G
    else:
        G_arr = np.asarray(G, dtype=float)
        G_use = np.where(np.isfinite(G_arr), G_arr, DEFAULT_SLOPE_G) if G_arr.ndim else (
            DEFAULT_SLOPE_G if not np.isfinite(G_arr) else float(G_arr))

    tan_half = np.tan(alpha / 2)
    phi1 = np.exp(-3.33 * tan_half ** 0.63)
    phi2 = np.exp(-1.87 * tan_half ** 1.22)
    phase_term = (1 - G_use) * phi1 + G_use * phi2
    phase_term = np.clip(phase_term, 1e-10, None)  # guard log10(0) right at alpha=180deg
    return H + 5 * np.log10(r_helio_au * delta_au) - 2.5 * np.log10(phase_term)


def _precompute_frame_geometry(frame_mjds, data_dir=None, allow_download=True):
    """TESS spacecraft barycentric position at each unique frame epoch --
    a pure function of time (spkpos against the epoch alone, plus Earth's
    barycentric position at that same epoch), identical regardless of
    which asteroid is being predicted. precise_ephemeris previously
    recomputed this independently for every (asteroid, frame) pair -- for
    a dense cut with hundreds of survivors sharing the same frame_mjds,
    that's hundreds of redundant identical SPICE/ASSIST lookups per frame.
    Computed once per cut in the main process and shared read-only across
    worker processes via the pool initializer, keyed by mjd.

    Deliberately excludes the light-time-corrected Sun position used
    inside precise_ephemeris's Newton loop (sun_em at t_em = t_target -
    light_time): light_time is asteroid-specific (depends on that
    object's own TESS distance), so t_em differs slightly per asteroid
    even at the "same" frame -- sharing a single Sun lookup there would
    trade this module's validated ~0.04 arcsec accuracy for speed, which
    isn't the trade being made here.
    """
    import spiceypy as sp
    from astropy.time import Time

    data_dir = data_dir or default_data_dir()
    ephem = load_assist_ephem(data_dir)
    jd_ref = ephem.jd_ref

    table = {}
    for mjd in np.unique(np.asarray(frame_mjds)):
        get_tess_kernel_for_epoch(mjd, data_dir, allow_download=allow_download)
        t_obs = Time(mjd, format="mjd", scale="utc")
        t_target_days = t_obs.tdb.jd - jd_ref
        et = sp.str2et(t_obs.isot)
        tess_off_km, _ = sp.spkpos("-95", et, "J2000", "NONE", "EARTH")
        tess_off_au = np.array(tess_off_km) / AU_KM
        earth_obs = ephem.get_particle("Earth", t_target_days)
        tess_bary_pos = np.array([earth_obs.x, earth_obs.y, earth_obs.z]) + tess_off_au
        table[float(mjd)] = (t_target_days, tess_bary_pos)
    return table


_frame_geometry_cache = None


def precise_ephemeris(mpcorb_row, frame_mjds, data_dir=None, allow_download=True, frame_geometry=None):
    """Full ASSIST-perturbed, light-time corrected TESS-relative RA/Dec for
    one object at every given frame time. Returns a DataFrame with one row
    per frame: mjd, ra, dec.

    Uses a single REBOUND simulation stepped forward through the frames in
    time order, rather than re-integrating from the object's MPCORB epoch
    (which can be years away from the observing window) on every frame --
    doing that naively meant every one of thousands of frames paid the
    full multi-year integration cost from scratch, which is the dominant
    cost of this whole module. IAS15 (REBOUND's default, non-symplectic)
    supports the small backward nudges the light-time correction needs
    without a full restart.

    Fetches whichever TESS SPK kernel segment covers each frame's epoch as
    it goes (via get_tess_kernel_for_epoch, itself cached) since each MAST
    segment only covers ~2 weeks and a full sector window can span more
    than one."""
    import assist
    import rebound
    import spiceypy as sp
    from astropy.time import Time
    from skyfield.functions import to_polar

    data_dir = data_dir or default_data_dir()
    ephem = load_assist_ephem(data_dir)
    jd_ref = ephem.jd_ref

    t_ref, helio_pos, helio_vel = _state_vector_at_epoch(mpcorb_row, data_dir)
    t_ref_days = t_ref.tdb - jd_ref
    sun_ref = ephem.get_particle("Sun", t_ref_days)
    bary_pos = helio_pos + np.array([sun_ref.x, sun_ref.y, sun_ref.z])
    bary_vel = helio_vel + np.array([sun_ref.vx, sun_ref.vy, sun_ref.vz])

    sim = rebound.Simulation()
    assist.Extras(sim, ephem)
    sim.t = t_ref_days
    sim.add(x=bary_pos[0], y=bary_pos[1], z=bary_pos[2],
            vx=bary_vel[0], vy=bary_vel[1], vz=bary_vel[2])

    order = np.argsort(frame_mjds)
    sorted_mjds = np.asarray(frame_mjds)[order]

    geometry = frame_geometry if frame_geometry is not None else _frame_geometry_cache

    rows = [None] * len(sorted_mjds)
    light_time = 0.0
    for out_i, mjd in zip(order, sorted_mjds):
        cached = geometry.get(float(mjd)) if geometry is not None else None
        if cached is not None:
            t_target_days, tess_bary_pos = cached
        else:
            get_tess_kernel_for_epoch(mjd, data_dir, allow_download=allow_download)
            t_obs = Time(mjd, format="mjd", scale="utc")
            t_target_days = t_obs.tdb.jd - jd_ref
            et = sp.str2et(t_obs.isot)
            tess_off_km, _ = sp.spkpos("-95", et, "J2000", "NONE", "EARTH")
            tess_off_au = np.array(tess_off_km) / AU_KM
            earth_obs = ephem.get_particle("Earth", t_target_days)
            tess_bary_pos = np.array([earth_obs.x, earth_obs.y, earth_obs.z]) + tess_off_au

        vec = None
        sun_vec = None
        for _ in range(4):
            t_em = t_target_days - light_time
            sim.integrate(t_em)
            p = sim.particles[0]
            obj_pos = np.array([p.x, p.y, p.z])
            vec = obj_pos - tess_bary_pos
            sun_em = ephem.get_particle("Sun", t_em)
            sun_vec = obj_pos - np.array([sun_em.x, sun_em.y, sun_em.z])  # Sun -> object
            light_time = np.linalg.norm(vec) * LIGHT_TIME_AU_DAYS

        _, dec_rad, ra_rad = to_polar(vec)
        delta = float(np.linalg.norm(vec))       # object-observer (TESS) distance, AU
        r_helio = float(np.linalg.norm(sun_vec))  # object-Sun distance, AU
        cos_alpha = np.dot(sun_vec, vec) / (r_helio * delta)
        phase_angle_deg = math.degrees(math.acos(np.clip(cos_alpha, -1.0, 1.0)))
        rows[out_i] = dict(mjd=mjd, ra=math.degrees(ra_rad) % 360, dec=math.degrees(dec_rad),
                            r_helio_au=r_helio, delta_au=delta, phase_angle_deg=phase_angle_deg)

    return pd.DataFrame(rows)


def _pool_worker_init(data_dir, frame_geometry=None):
    """Pre-load the ~1GB ASSIST ephemeris and SPICE leapseconds kernel once
    per worker process, so each of the (many) precise_ephemeris calls a
    worker handles reuses them instead of reloading from disk every time.
    Also caches the shared per-frame TESS/Earth geometry table (see
    _precompute_frame_geometry) computed once in the main process, so
    every survivor this worker handles reuses it instead of recomputing
    the same time-only SPICE lookups redundantly."""
    global _frame_geometry_cache
    load_assist_ephem(data_dir)
    furnish_spice_generic(data_dir)
    if frame_geometry is not None:
        _frame_geometry_cache = frame_geometry


def _precise_ephemeris_worker(row, frame_mjds, data_dir, allow_download):
    return precise_ephemeris(row, frame_mjds, data_dir, allow_download=allow_download)


# ---------------------------------------------------------------------------
# Top-level: predict every catalogued asteroid crossing a footprint
# ---------------------------------------------------------------------------

def predict_asteroids_for_footprint(ra_center_deg, dec_center_deg, radius_deg,
                                       mjd_start, mjd_end, frame_mjds, wcs,
                                       n_coarse_samples=None, faint_limit_mag=20.8,
                                       data_dir=None, plot_path=None, allow_download=True,
                                       sector_snapshot_path=None):
    """Predict every MPCORB-catalogued object crossing a circular footprint
    (ra_center, dec_center, radius) at any point in [mjd_start, mjd_end],
    and build a precise per-frame ephemeris (ra, dec, x, y, flux, mag where
    measurable) for each, restricted to the frames where it's actually
    within the footprint.

    `wcs` is used only to convert precise sky positions back to pixel
    coordinates for frames within the footprint (an astropy-compatible WCS
    object, e.g. from tesswcs.WCS.from_sector or the cut's own archived
    solution). `frame_mjds` order defines the "frame number" (its index)
    used for the trail plot's colour axis, so pass it in true frame order.
    If plot_path is given, saves a figure of every surviving object's
    track through the footprint, coloured by frame number.

    allow_download controls whether get_tess_kernel_for_epoch is allowed
    to fetch missing TESS SPK kernels from MAST on demand. Set False (the
    Tessellate SLURM stage always does) when running on a compute node
    with no internet access -- verify_data_available is then run up front
    instead, so a missing/uncovered file fails fast with a clear message
    rather than partway through a worker process. Leave True for
    interactive/login-node use, where on-demand downloading is fine.

    sector_snapshot_path: a per-sector re-epoched catalogue from
    build_sector_mpcorb_snapshot, used in place of the raw MPCORB.DAT when given. Use this for
    any sector whose observing window is not close in time to MPCORB.DAT's own catalogue
    epoch -- reprocessing an old sector with today's file, for example -- since coarse
    unperturbed propagation across a multi-year gap can misplace an object by many degrees
    (confirmed on (1) Ceres: 15 deg, from a 2112-day gap). None keeps the original raw-MPCORB
    behaviour unchanged, correct exactly when this module's docstring assumption holds: the
    observing window is close to MPCORB.DAT's own epoch already.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    data_dir = data_dir or default_data_dir()
    if not allow_download:
        verify_data_available(data_dir, frame_mjds=frame_mjds)
    if sector_snapshot_path is not None:
        mpcorb = load_mpcorb_snapshot(sector_snapshot_path)
        epoch_mjd = mpcorb["epoch_mjd"].values
    else:
        mpcorb = load_mpcorb(data_dir)
        epoch_mjd = _mpc_packed_epoch_to_mjd(mpcorb["epoch_packed"].values)

    # Coarse sample epochs across the observing window -- hoisted above stage 1 (it used to
    # sit between stage 1b and stage 2) because stage 1 now needs it too: both stages
    # propagate the same way, at the same epochs, just over different-sized candidate sets.
    max_motion_deg_per_day = 1.5
    if n_coarse_samples is None:
        # pick enough samples that the between-sample motion margin stays
        # comparable to the footprint radius itself, rather than dwarfing
        # it -- this stage is cheap (vectorized Kepler propagation over
        # the whole catalogue) so there's no reason to under-sample and
        # pass a huge, slow-to-refine candidate list into stage 2/precise
        window_days = mjd_end - mjd_start
        target_spacing = max(radius_deg / max_motion_deg_per_day, window_days / 500)
        n_coarse_samples = int(np.clip(np.ceil(window_days / target_spacing) + 1, 5, 500))
    sample_mjds = np.linspace(mjd_start, mjd_end, n_coarse_samples)
    sample_spacing = sample_mjds[1] - sample_mjds[0] if n_coarse_samples > 1 else (mjd_end - mjd_start)
    coarse_margin = radius_deg + max_motion_deg_per_day * sample_spacing

    ecl = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg).barycentrictrueecliptic
    stage1 = ecliptic_reachable_mask(mpcorb, epoch_mjd, sample_mjds, ecl.lat.deg,
                                     margin_deg=coarse_margin)
    print(f"  Stage 1 (ecliptic reachability): {stage1.sum()} / {len(mpcorb)} survive", flush=True)

    stage1b_input = mpcorb[stage1].reset_index(drop=True)
    stage1b = brightness_reachable_mask(stage1b_input, faint_limit_mag=faint_limit_mag)
    print(f"  Stage 1b (best-case brightness <= {faint_limit_mag}): "
          f"{stage1b.sum()} / {len(stage1b_input)} survive", flush=True)

    stage2_input = stage1b_input[stage1b].reset_index(drop=True)
    # margin: footprint radius + typical geocentric motion budget between consecutive coarse
    # samples (NOT the full window -- an object can only drift so far between adjacent
    # samples, however long the overall window is). Same coarse_margin stage 1 uses above.
    stage2, hit_samples_by_row, rate_by_row = coarse_position_mask(
        stage2_input, ra_center_deg, dec_center_deg, radius_deg + coarse_margin, sample_mjds,
        return_hit_samples=True)
    survivors = stage2_input[stage2].reset_index(drop=True)
    print(f"  Stage 2 (coarse position): {len(survivors)} / {len(stage2_input)} survive", flush=True)

    mjd_to_frame = {mjd: i for i, mjd in enumerate(frame_mjds)}

    # Per survivor, restrict the expensive precise ephemeris to frames near where the
    # coarse stage actually saw it close to the footprint, instead of every frame in the
    # observing window -- most survivors only cross the footprint briefly, but previously
    # paid the full per-frame ASSIST/light-time cost for every frame regardless (see
    # module docstring).
    #
    # Padding by a flat sample_spacing is NOT safe: the margin above (and sample_spacing
    # itself) is sized against max_motion_deg_per_day, but real close near-Earth objects
    # can move several times faster -- caught empirically (a real (348) May/(595) Polyxena-
    # style validation run against this exact code found two NEOs at ~0.05 AU moving up to
    # 4.9 deg/day, well past the 1.5 deg/day assumption, whose frames a flat pad silently
    # dropped relative to the unrestricted baseline). Instead scale the pad by how much
    # faster THIS object's own observed rate (rate_by_row, from coarse_position_mask) is
    # than that assumption, plus a 2x safety factor since the rate is only an estimate from
    # discrete coarse samples, not a true bound. Falls back to the full frame list if
    # narrowing somehow leaves nothing -- must never silently drop a real detection.
    frame_mjds_arr = np.asarray(frame_mjds, dtype=float)
    survivor_orig_idx = np.where(stage2)[0]
    narrowed_frames = []
    for orig_i in survivor_orig_idx:
        hit_t = hit_samples_by_row.get(int(orig_i))
        rate = rate_by_row.get(int(orig_i), max_motion_deg_per_day)
        pad = sample_spacing * max(1.0, rate / max_motion_deg_per_day) * 2.0
        mask = np.zeros(len(frame_mjds_arr), dtype=bool)
        if hit_t is not None and len(hit_t):
            for t in hit_t:
                mask |= (frame_mjds_arr >= t - pad) & (frame_mjds_arr <= t + pad)
        subset = frame_mjds_arr[mask]
        narrowed_frames.append(subset if len(subset) else frame_mjds_arr)

    all_rows = []
    if len(survivors):
        # Shared per-frame TESS/Earth geometry (see _precompute_frame_geometry) -- a pure
        # function of time, identical for every survivor, computed once here instead of
        # redundantly inside precise_ephemeris for each of them.
        frame_geometry = _precompute_frame_geometry(frame_mjds, data_dir, allow_download)

        # ProcessPoolExecutor rejects max_workers=0 -- a cut with zero stage-2 survivors (a real,
        # common case for high-ecliptic-latitude footprints) would otherwise crash here instead
        # of legitimately returning an empty result
        n_workers = min(len(survivors), max(1, (os.cpu_count() or 4) - 1))
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_worker_init,
                                  initargs=(data_dir, frame_geometry)) as pool:
            futures = {pool.submit(_precise_ephemeris_worker, row, narrowed_frames[k], data_dir, allow_download): row
                       for k, (_, row) in enumerate(survivors.iterrows())}
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    eph = fut.result()
                except Exception as ex:
                    print(f"  {row.designation}: precise ephemeris failed: {ex}", flush=True)
                    continue
                sky = SkyCoord(ra=eph["ra"].values * u.deg, dec=eph["dec"].values * u.deg)
                sep = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg).separation(sky).deg
                in_fov = sep <= radius_deg
                if not in_fov.any():
                    continue
                eph = eph[in_fov].copy()
                x, y = wcs.world_to_pixel(sky[in_fov])
                eph["x"], eph["y"] = x, y
                eph["frame"] = eph["mjd"].map(mjd_to_frame)
                eph["designation"] = row.designation
                eph["magnitude_H"] = row.magnitude_H
                eph["magnitude_G"] = row.magnitude_G
                eph["mag_expected"] = expected_apparent_magnitude(
                    row.magnitude_H, row.magnitude_G, eph["r_helio_au"].values,
                    eph["delta_au"].values, eph["phase_angle_deg"].values)
                all_rows.append(eph)

    if not all_rows:
        result = pd.DataFrame(columns=["designation", "mjd", "frame", "ra", "dec", "x", "y",
                                        "r_helio_au", "delta_au", "phase_angle_deg",
                                        "magnitude_H", "magnitude_G", "mag_expected"])
    else:
        result = pd.concat(all_rows, ignore_index=True)

    if plot_path is not None:
        plot_asteroid_trails(result, plot_path)

    return result


def _mag_to_alpha(mag, bright_ref=14.0, faint_limit=20.8, alpha_min=0.12, alpha_max=1.0):
    """Linear-in-magnitude alpha scaling: objects at/brighter than
    bright_ref are fully opaque, objects at TESS's faint limit fade to
    alpha_min, so a busy field reads at a glance which tracks are worth
    a look and which are marginal. Unknown magnitude (NaN) gets alpha_max
    rather than vanishing, so missing-H objects stay visible."""
    mag = np.asarray(mag, dtype=float)
    frac = np.clip((mag - bright_ref) / (faint_limit - bright_ref), 0, 1)
    alpha = alpha_max - frac * (alpha_max - alpha_min)
    return np.where(np.isfinite(mag), alpha, alpha_max)


def plot_asteroid_trails(ephemeris_df, save_path, footprint_size=None):
    """Every predicted asteroid's track through the footprint, in pixel
    coordinates, points coloured by frame number (a single shared colour
    axis across all objects, so relative timing between different tracks
    is visible too, not just motion within one track), with marker/line
    opacity scaled by each object's expected apparent brightness (mag_expected,
    if present) so faint, marginal tracks visually recede against bright,
    easily-recovered ones instead of all reading with equal visual weight.

    footprint_size, if given, is the full pixel side length of the (square)
    cutout -- axis limits are pinned to [0, footprint_size] on both axes so
    the plotted box is actually square and matches the true field, rather
    than autoscaling to wherever the surviving tracks happen to fall
    (aspect='equal' alone only equalises the x/y *scale*, not the box
    shape, so a data extent that's naturally taller than wide would still
    render as a non-square rectangle without this).

    Tracks are drawn as a single LineCollection rather than one ax.plot()
    call per object -- a dense field can have 1000+ surviving tracks, and
    that many separate Line2D artists was the dominant memory allocator in
    this whole module by a wide margin (memray profiling on a real dense
    Sector 44 cut: 6.3GB / 7.36M individual allocations from matplotlib's
    per-artist draw path, for what should be one batched draw call).
    No per-track text labels -- with a dense field routinely surviving
    into the hundreds or thousands of tracks, any labeling scheme (tried:
    every track, brightest N, a magnitude/numbered-object cut) still
    either clutters the figure illegibly or reads as an arbitrary partial
    list; the colour-by-frame trail + brightness-by-alpha encoding already
    carries the useful information without needing designations spelled
    out on the plot itself."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(7, 7))
    if footprint_size is not None:
        ax.set_xlim(0, footprint_size)
        ax.set_ylim(0, footprint_size)
    # x/y are both in pixel units of the same footprint -- an unequal aspect would stretch
    # one axis relative to the other and misrepresent the field as non-square
    ax.set_aspect("equal", adjustable="box")
    if len(ephemeris_df) == 0:
        ax.text(0.5, 0.5, "No asteroids predicted in this footprint",
                 ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    else:
        has_mag = "mag_expected" in ephemeris_df.columns
        vmin, vmax = ephemeris_df["frame"].min(), ephemeris_df["frame"].max()

        segments, line_alphas = [], []
        for designation, track in ephemeris_df.groupby("designation"):
            track = track.sort_values("frame")
            med_mag = float(track["mag_expected"].median()) if has_mag else np.nan
            segments.append(np.column_stack([track["x"].values, track["y"].values]))
            line_alphas.append(_mag_to_alpha(med_mag) if has_mag else 0.75)

        lc = LineCollection(segments, colors="0.75", linewidths=0.8, zorder=1)
        lc.set_alpha(line_alphas)
        ax.add_collection(lc)

        cmap = plt.get_cmap("cividis")
        norm = plt.Normalize(vmin, vmax)
        colors = cmap(norm(ephemeris_df["frame"].values))
        colors[:, 3] = _mag_to_alpha(ephemeris_df["mag_expected"].values) if has_mag else 1.0
        ax.scatter(ephemeris_df["x"], ephemeris_df["y"], color=colors, s=10, zorder=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # attach via the axes' own divider (not fig.colorbar's default heuristic) so the
        # colorbar tracks the axes' actual box height post-aspect-lock, not the full figure
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
        fig.colorbar(sm, cax=cax, label="frame number")

        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title(f"{ephemeris_df['designation'].nunique()} asteroid track(s) in footprint "
                      "(opacity ~ expected brightness)" if has_mag else "")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

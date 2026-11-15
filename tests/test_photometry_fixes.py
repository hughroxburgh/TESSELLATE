"""Checks of the 2026-11-15 asteroid photometry changes (asteroid-photometry-leak branch).
  median       _nanmedian_rows equals np.nanmedian(axis=1) exactly (NaN patterns, all-NaN rows, odd / even counts)
  regression   forced_psf_photometry (frame shifts, background leak, trailed rows) equals the version before the changes
  stamps       _aligned_stamps equals the loop of ndimage.shift(stamp, (-dy, -dx), order=1, mode='nearest') it replaces, and
               measure_stack_centroid_offset gives the version-before's offsets, significances and fluxes
  positions    smooth_track_positions on noisy injections through a simulated alignment (sources at their raw-frame
               positions, every frame resampled with ndimage.shift(order=5, mode='nearest') and a random shift): a bright
               trailed track whose true position drifts from its ephemeris (smooth, sub-0.1 px) has the drift recovered
               and its flux scatter about the truth reduced; a faint and a very faint track (per-frame S/N ~3, detected only
               when stacked) with no drift are left at their ephemerides (no noise-driven wandering)
Run: python3 tests/test_photometry_fixes.py <PRF path> [<git ref before the changes>]"""
import os, subprocess, sys, types  # noqa: E401
import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift

here = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(here)
sys.path.insert(0, repo)
from tessellate import asteroid_photometry as ap  # noqa: E402
from tessellate.prf_grid import get_grid  # noqa: E402

prf_path = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else "b6947d3"
cam, ccd, sector, X0, Y0 = 4, 4, 43, 700.0, 900.0
rng = np.random.default_rng(11)
src = subprocess.run(["git", "-C", repo, "show", f"{ref}:tessellate/asteroid_photometry.py"], capture_output=True, text=True, check=True).stdout
old = types.ModuleType(f"tessellate.asteroid_photometry_{ref}"); old.__package__ = "tessellate"
exec(compile(src, f"{ref}:asteroid_photometry.py", "exec"), old.__dict__)

# median
a = rng.normal(size=(500, 81))
a[rng.random(a.shape) < 0.2] = np.nan
a[:5] = np.nan
a[5:10, 1:] = np.nan
a[10:20, 40:] = np.nan
with np.errstate(all="ignore"):
    import warnings
    warnings.simplefilter("ignore")
    m0 = np.nanmedian(a, axis=1)
m1 = ap._nanmedian_rows(a)
same = np.array_equal(m0, m1, equal_nan=True)
print(f"median: _nanmedian_rows == np.nanmedian on {len(a)} rows (NaN patterns incl. all-NaN): {same}")
assert same

# regression of the photometry
cube = (rng.normal(0, 5, (30, 70, 70)) + 30).astype(np.float32)
bkg = (rng.normal(0, 1, cube.shape) + 3).astype(np.float32)
S = rng.uniform(-0.3, 0.3, (len(cube), 2))
t = np.arange(30) * 600 / 86400.0
eph = pd.DataFrame(dict(designation=np.repeat(["a", "b", "c"], 30), frame=np.tile(np.arange(30), 3), mjd=np.tile(t, 3),
                        x=np.r_[14 + 0.6 * np.arange(30), 12 + 1.5 * np.arange(30), 35 + 0.02 * np.arange(30)],
                        y=np.r_[30 + 0.1 * np.arange(30), 12 + 1.5 * np.arange(30), 50 - 0.05 * np.arange(30)]))
for kw in (dict(), dict(exptime_s=600.0), dict(exptime_s=600.0, shifts=S, background=bkg)):
    p0 = old.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path, **kw)
    p1 = ap.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path, **kw)
    d = {c: np.nanmax(np.abs(p1[c].to_numpy() - p0[c].to_numpy()) / np.nanmax(np.abs(p0[c].to_numpy()))) if np.isfinite(p0[c]).any() else 0.0
         for c in ("flux", "e_flux", "background", "leak")}
    print(f"regression {sorted(kw)}: {len(p1)} rows, max relative difference " + ", ".join(f"{c} {v:.1e}" for c, v in d.items()))
    assert all(v < 1e-12 for v in d.values())

# stamps and the stacked offset fit
x = rng.uniform(8, 62, 300); y = rng.uniform(8, 62, 300); fr = rng.integers(0, 30, 300)
x[:4] = [6.0, 6.5, 63.5, 64.0]                   # rows at and beyond the stamp's reach of the cube edge
track = pd.DataFrame(dict(x=x, y=y, frame=fr))
loop = []
for row in track.itertuples():
    xi, yi = int(round(row.x)), int(round(row.y))
    if xi - 6 < 0 or yi - 6 < 0 or xi + 7 > cube.shape[2] or yi + 7 > cube.shape[1]:
        continue
    loop.append(nd_shift(cube[row.frame, yi - 6:yi + 7, xi - 6:xi + 7], (-(row.y - yi), -(row.x - xi)), order=1, mode="nearest"))
loop = np.array(loop)
new = ap._aligned_stamps(track, cube, 6)
dd = np.abs(new.astype(float) - loop.astype(float)).max()
print(f"stamps: {len(new)} = {len(loop)} stamps, dtype {new.dtype}; max |difference| from the ndimage.shift loop {dd:.1e} "
      f"(float32 resolution here ~{np.spacing(np.float32(60)):.0e})")
assert len(new) == len(loop) and new.dtype == loop.dtype and dd <= 4 * np.spacing(np.float32(60))
grid = get_grid(cam, ccd, sector, prf_path)
for amp in (300.0, 3000.0):
    cubei = (rng.normal(0, 5, (40, 60, 60)) + 20).astype(np.float32)
    xs, ys = 20 + 0.4 * np.arange(40), 25 + 0.15 * np.arange(40)
    for k_ in range(40):
        xi, yi = int(round(xs[k_] + 0.3)), int(round(ys[k_] - 0.2))
        cubei[k_, yi - 8:yi + 9, xi - 8:xi + 9] += amp * grid.images([X0 + xs[k_]], [Y0 + ys[k_]], [xs[k_] + 0.3 - xi], [ys[k_] - 0.2 - yi], 8)[0]
    tr = pd.DataFrame(dict(designation="z", x=xs, y=ys, frame=np.arange(40), mjd=np.arange(40) * 600 / 86400.0))
    r0 = old.measure_stack_centroid_offset(tr, cubei, sector, cam, ccd, X0, Y0, prf_path=prf_path)
    r1 = ap.measure_stack_centroid_offset(tr, cubei, sector, cam, ccd, X0, Y0, prf_path=prf_path)
    diff = max(abs(a_ - b_) / max(abs(a_), 1e-9) for a_, b_ in zip(r0[:4], r1[:4]))
    print(f"stacked offset fit, amplitude {amp:.0f}: before {np.round(r0[:4], 5).tolist()}, now {np.round(r1[:4], 5).tolist()}; "
          f"max relative difference {diff:.1e}")
    assert diff < 1e-6

# smooth positions
F_BRIGHT, F_FAINT, F_VFAINT, SKY, NOISE, nfr, H = 40000.0, 400.0, 60.0, 100.0, 5.0, 300, 15
S = rng.uniform(-0.1, 0.1, (nfr, 2))
tau = np.linspace(-1, 1, nfr)
specs = {"bright": (F_BRIGHT, 30.0, 40.0, 0.25, 0.02, 0.03 + 0.03 * tau, -0.04 + 0.06 * tau ** 2),
         "faint": (F_FAINT, 30.0, 110.0, 0.25, 0.02, 0.0 * tau, 0.0 * tau),
         "very faint": (F_VFAINT, 20.0, 75.0, 0.3, 0.0, 0.0 * tau, 0.0 * tau)}
ny, nx = 150, 140
frames = np.full((nfr, ny, nx), SKY)
truth_pos, rows = {}, []
for name, (F, x0, y0, vx, vy, ddx, ddy) in specs.items():
    xe, ye = x0 + vx * np.arange(nfr), y0 + vy * np.arange(nfr)            # ephemeris (aligned frame)
    xt, yt = xe + ddx, ye + ddy                                            # truth
    truth_pos[name] = (xe, ye, xt, yt, F)
    rows.append(pd.DataFrame(dict(designation=name, frame=np.arange(nfr), mjd=np.arange(nfr) * 600 / 86400.0, x=xe, y=ye)))
cube = np.empty((nfr, ny, nx), dtype=np.float32)
for k_ in range(nfr):
    raw = frames[k_].copy()
    for name, (xe, ye, xt, yt, F) in truth_pos.items():
        rx, ry = xt[k_] - S[k_, 1], yt[k_] - S[k_, 0]
        rxi, ryi = int(round(rx)), int(round(ry))
        L = np.hypot(specs[name][3], specs[name][4])
        ss = np.linspace(-0.5, 0.5, int(np.ceil(L / 0.02)) + 1)
        qx, qy = rx + ss * specs[name][3], ry + ss * specs[name][4]
        raw[ryi - H:ryi + H + 1, rxi - H:rxi + H + 1] += F * grid.images(X0 + qx, Y0 + qy, qx - rxi, qy - ryi, H).mean(0)
    cube[k_] = (nd_shift(raw, S[k_], order=5, mode="nearest") + rng.normal(0, NOISE, (ny, nx))).astype(np.float32)
eph = pd.concat(rows, ignore_index=True)
refined, diag = ap.smooth_track_positions(cube, eph, sector, cam, ccd, X0, Y0, list(specs), prf_path=prf_path, exptime_s=600.0, shifts=S)
print("positions: " + "; ".join(f"{r.designation} degree x {r.degree_x}, y {r.degree_y}, offset rms {r.offset_rms_x:.4f}, {r.offset_rms_y:.4f} px"
                                for r in diag.itertuples()))
for name, (xe, ye, xt, yt, F) in truth_pos.items():
    m = (refined.designation == name).to_numpy()
    rx_, ry_ = refined.x.to_numpy()[m], refined.y.to_numpy()[m]
    err0 = np.sqrt(np.mean((xe - xt) ** 2 + (ye - yt) ** 2))
    err1 = np.sqrt(np.mean((rx_ - xt) ** 2 + (ry_ - yt) ** 2))
    e_ = eph[eph.designation == name]
    f0 = ap.forced_psf_photometry(cube, e_, sector, cam, ccd, X0, Y0, prf_path=prf_path, exptime_s=600.0, shifts=S).sort_values("frame")
    f1 = ap.forced_psf_photometry(cube, refined[refined.designation == name], sector, cam, ccd, X0, Y0, prf_path=prf_path, exptime_s=600.0,
                                  shifts=S).sort_values("frame")
    tru = np.array([F * grid.images([X0 + xt[k_]], [Y0 + yt[k_]], [xt[k_] - round(xt[k_])], [yt[k_] - round(yt[k_])], 4)[0].sum() for k_ in range(nfr)])
    s0 = np.std(f0.flux.to_numpy() / tru[f0.frame.to_numpy()] - 1) * 100
    s1 = np.std(f1.flux.to_numpy() / tru[f1.frame.to_numpy()] - 1) * 100
    print(f"  {name}: position error rms {err0:.4f} px at the ephemeris -> {err1:.4f} px refined; flux / truth scatter {s0:.3f} -> {s1:.3f} %")
    if name == "bright":
        assert err1 < 0.3 * err0 and s1 < s0, "bright drift not recovered"
    else:
        assert err1 < 0.02, "faint track moved off its (correct) ephemeris"
print("all checks passed")

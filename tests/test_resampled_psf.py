"""Checks of the resampled PSF template in asteroid_photometry.forced_psf_photometry (shifts=; asteroid-photometry-leak branch).
  operator    M(sy) R M(sx)^T reproduces ndimage.shift(R, (sy, sx), order=5, mode='nearest') on random stamps
  regression  shifts=None reproduces the committed branch version exactly (point and trailed rows, flux and leak);
              all-zero shifts give the same fluxes
  injection   the reduction simulated: a source (point, or trailed over L px per exposure) at its raw-frame position on a
              flat sky, the whole frame resampled with ndimage.shift(frame, (sy, sx), order=5, mode='nearest') and a random
              shift per frame, then fitted at its aligned position with and without shifts. True flux in the point-source
              convention (flux x the point PRF's sum in the 9 x 9 stamp at the aligned position). Noise-free, so the
              spread is the template's error alone
  timing      a cut-sized row set with and without shifts
Run: python3 tests/test_resampled_psf.py <PRF path> [<git ref of the version to regress against>]"""
import os, subprocess, sys, time, types  # noqa: E401
import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift

here = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(here)
sys.path.insert(0, repo)
from tessellate import asteroid_photometry as ap  # noqa: E402
from tessellate.prf_grid import get_grid  # noqa: E402

prf_path = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else "ce509e2"
cam, ccd, sector, X0, Y0 = 4, 4, 43, 700.0, 900.0
rng = np.random.default_rng(5)

# operator
for n in (21, 31):
    R = rng.normal(size=(6, n, n)); s = rng.uniform(-0.6, 0.6, (6, 2))
    a = np.array([nd_shift(R[i], s[i], order=5, mode="nearest") for i in range(len(R))])
    b = ap._shift_operators(s[:, 0], n) @ R @ ap._shift_operators(s[:, 1], n).transpose(0, 2, 1)
    d = np.abs(a - b).max()
    print(f"operator, {n} x {n} stamps: max |difference| from ndimage.shift {d:.1e}")
    assert d < 1e-10

# regression: two tracks, one slow (0.6 px per exposure, point rows) and one trailed (3 px per exposure, diagonal)
src = subprocess.run(["git", "-C", repo, "show", f"{ref}:tessellate/asteroid_photometry.py"], capture_output=True, text=True, check=True).stdout
ref_mod = types.ModuleType("tessellate.asteroid_photometry_ref"); ref_mod.__package__ = "tessellate"
exec(compile(src, f"{ref}:asteroid_photometry.py", "exec"), ref_mod.__dict__)
cube = rng.normal(0, 5, (20, 64, 64)) + 30
bkg = rng.normal(0, 1, cube.shape) + 3
t = np.arange(20) * 600 / 86400.0
eph = pd.DataFrame(dict(designation=np.repeat(["a", "b"], 20), frame=np.tile(np.arange(20), 2), mjd=np.tile(t, 2),
                        x=np.r_[14 + 0.6 * np.arange(20), 12 + 2.1 * np.arange(20)], y=np.r_[30 + 0.1 * np.arange(20), 12 + 2.1 * np.arange(20)]))
for exptime in (None, 600.0):
    a = ref_mod.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path, background=bkg, exptime_s=exptime)
    b = ap.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path, background=bkg, exptime_s=exptime)
    z = ap.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path, background=bkg, exptime_s=exptime,
                                 shifts=np.zeros((len(cube), 2)))
    d = max(np.max(np.abs(a.flux.to_numpy() - b.flux.to_numpy())), np.max(np.abs(a.leak.to_numpy() - b.leak.to_numpy())))
    dz = np.max(np.abs(z.flux.to_numpy() - b.flux.to_numpy()) / np.abs(b.flux.to_numpy()).max())
    print(f"regression ({'trailed' if exptime else 'point'}) vs {ref}: {len(a)} rows, max |flux, leak difference| {d:.1e}; "
          f"zero shifts vs none: max relative flux difference {dz:.1e}")
    assert d < 1e-9 and dz < 1e-9

# injection
grid = get_grid(cam, ccd, sector, prf_path)
F, SKY, nfr, HS = 10000.0, 100.0, 60, 15


def source_image(px, py, L, ang, xi, yi):
    """The source's image in the (2 HS + 1)^2 stamp about (xi, yi): the PRF averaged along the path in 0.02 px steps."""
    ss = np.linspace(-0.5, 0.5, int(np.ceil(L / 0.02)) + 1) if L > 0 else np.zeros(1)
    qx, qy = px + ss * L * np.cos(ang), py + ss * L * np.sin(ang)
    return grid.images(X0 + qx, Y0 + qy, qx - xi, qy - yi, HS).mean(0)


print(f"{'L px':>5s} {'direction':>9s} {'no shifts: median, rms, max |r - 1|':>38s} {'shifts: median, rms, max |r - 1|':>36s}")
for L, ang in ((0.0, 0.0), (0.5, 0.0), (2.0, np.pi / 4), (5.0, 0.0), (5.0, np.pi / 4)):
    if L == 0:          # static sources, one per frame at a random sub-pixel phase
        xs, ys = 40 + rng.uniform(-0.5, 0.5, nfr), 40 + rng.uniform(-0.5, 0.5, nfr)
        des = [f"s{k_}" for k_ in range(nfr)]
    else:               # one track moving L px per exposure
        xs = 30 + rng.uniform(-0.5, 0.5) + L * np.cos(ang) * np.arange(nfr)
        ys = 30 + rng.uniform(-0.5, 0.5) + L * np.sin(ang) * np.arange(nfr)
        des = ["z"] * nfr
    S = rng.uniform(-0.5, 0.5, (nfr, 2))                 # (dy, dx) per frame, as tessreduce's shift
    ny_, nx_ = int(ys.max()) + 40, int(xs.max()) + 40
    cubei = np.empty((nfr, ny_, nx_))
    truth = np.zeros(nfr)
    for k_ in range(nfr):
        rx, ry = xs[k_] - S[k_, 1], ys[k_] - S[k_, 0]     # raw-frame position
        rxi, ryi = int(round(rx)), int(round(ry))
        raw = np.full((ny_, nx_), SKY)
        raw[ryi - HS:ryi + HS + 1, rxi - HS:rxi + HS + 1] += F * source_image(rx, ry, L, ang, rxi, ryi)
        cubei[k_] = nd_shift(raw, S[k_], order=5, mode="nearest")
        xi_, yi_ = int(round(xs[k_])), int(round(ys[k_]))
        truth[k_] = F * grid.images([X0 + xs[k_]], [Y0 + ys[k_]], [xs[k_] - xi_], [ys[k_] - yi_], 4)[0].sum()
    ephi = pd.DataFrame(dict(designation=des, frame=np.arange(nfr), mjd=np.arange(nfr) * 600 / 86400.0, x=xs, y=ys))
    exptime = 600.0 if L > 0 else None
    out = []
    for sh in (None, S):
        fr = ap.forced_psf_photometry(cubei, ephi, sector, cam, ccd, X0, Y0, prf_path=prf_path, exptime_s=exptime, shifts=sh)
        fr = fr.sort_values("frame")
        r = fr.flux.to_numpy() / truth[fr.frame.to_numpy()]
        out.append(r)
    (r0, r1) = out
    print(f"{L:5.1f} {('x' if ang == 0 else 'diagonal'):>9s} {np.median(r0):14.4f} {np.std(r0):9.4f} {np.abs(r0 - 1).max():9.4f}"
          f" {np.median(r1):16.4f} {np.std(r1):9.4f} {np.abs(r1 - 1).max():9.4f}")
    if L == 0:
        assert np.abs(r1 - 1).max() < 1e-3, "resampled point template does not recover the injected flux"
    else:
        assert abs(np.median(r1) - 1) < 0.01 and np.std(r1) < 0.01, "resampled trailed template does not recover the injected flux"
print("injection passed")

# timing: 150 tracks x up to 500 frames on a 200 x 200 cube, trailed (0.3-2 px per exposure)
nfr, ntr = 500, 150
cube = rng.normal(0, 5, (nfr, 200, 200)) + 30
x0_, y0_ = rng.uniform(30, 60, ntr), rng.uniform(30, 170, ntr)
v_ = rng.uniform(0.3, 2.0, ntr)
eph = pd.DataFrame(dict(designation=np.repeat(np.arange(ntr).astype(str), nfr), frame=np.tile(np.arange(nfr), ntr),
                        mjd=np.tile(np.arange(nfr) * 600 / 86400.0, ntr),
                        x=(x0_[:, None] + v_[:, None] * np.arange(nfr)[None]).ravel(), y=np.repeat(y0_, nfr)))
eph = eph[eph.x < 190]
S = rng.uniform(-0.5, 0.5, (nfr, 2))
for sh in (None, S):
    t0 = time.perf_counter()
    fr = ap.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path, exptime_s=600.0, shifts=sh)
    print(f"timing, {'shifts' if sh is not None else 'no shifts'}: {len(fr)} rows in {time.perf_counter() - t0:.1f} s")

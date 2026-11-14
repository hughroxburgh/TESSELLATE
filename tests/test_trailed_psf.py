"""Checks of the trailed PSF template in asteroid_photometry.forced_psf_photometry (asteroid-photometry-leak branch).
  regression  trail off (no exposure time) reproduces the point-source fit of the committed branch version exactly
  injection   a source trailed over L px (the same PRF model averaged along the path, flat sky + Gaussian noise) is
              recovered at its injected flux (point-source convention) with the trailed template for every L, while
              the point template loses flux as L grows
Run: python3 tests/test_trailed_psf.py <PRF path> [<git ref of the version to regress against>]"""
import importlib.util, os, subprocess, sys, tempfile  # noqa: E401
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(here)
sys.path.insert(0, repo)
from tessellate import asteroid_photometry as ap  # noqa: E402
from tessellate.prf_grid import get_grid  # noqa: E402

prf_path = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else "9e37629"
cam, ccd, sector, X0, Y0 = 4, 4, 44, 700.0, 900.0
rng = np.random.default_rng(3)
# regression: random positions in a noise cube, trail off, against the committed version's function
cube = rng.normal(0, 5, (40, 64, 64)) + 30
eph = pd.DataFrame(dict(designation=np.repeat(["a", "b"], 20), frame=np.tile(np.arange(20), 2) * 2, mjd=np.tile(np.arange(20), 2) * 600 / 86400.0,
                        x=rng.uniform(10, 54, 40), y=rng.uniform(10, 54, 40)))
src = subprocess.run(["git", "-C", repo, "show", f"{ref}:tessellate/asteroid_photometry.py"], capture_output=True, text=True, check=True).stdout
with tempfile.TemporaryDirectory() as td:
    pk = os.path.join(td, "tessellate"); os.makedirs(pk)
    for fn in ("__init__.py", "prf_grid.py", "psf_flux_calibration.py", "tools.py"):
        if os.path.exists(os.path.join(repo, "tessellate", fn)):
            open(os.path.join(pk, fn), "w").write(open(os.path.join(repo, "tessellate", fn)).read() if fn != "__init__.py" else "")
    open(os.path.join(pk, "asteroid_photometry.py"), "w").write(src)
    spec = importlib.util.spec_from_file_location("tessellate_ref.asteroid_photometry", os.path.join(pk, "asteroid_photometry.py"),
                                                  submodule_search_locations=None)
    sys.path.insert(0, td)
    import importlib
    old = importlib.import_module("tessellate.asteroid_photometry") if False else None
    ref_mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("tessellate.asteroid_photometry_ref", os.path.join(pk, "asteroid_photometry.py")))
    ref_mod.__package__ = "tessellate"
    importlib.util.spec_from_file_location("tessellate.asteroid_photometry_ref", os.path.join(pk, "asteroid_photometry.py")).loader.exec_module(ref_mod)
    a = ref_mod.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path)
b = ap.forced_psf_photometry(cube, eph, sector, cam, ccd, X0, Y0, prf_path=prf_path)
d = np.max(np.abs(a.flux.to_numpy() - b.flux.to_numpy()))
print(f"regression (trail off) vs {ref}: {len(a)} rows, max |flux difference| {d:.3e}; trail_px all zero: {bool((b.trail_px == 0).all())}")
assert d < 1e-9
# injection: one trailed source per frame, moving uniformly at L px per exposure along +x or the diagonal, true flux
# 10,000 e-/s in the point-source convention (flux x the point PRF's sum in the 9x9 stamp at that frame's position)
grid = get_grid(cam, ccd, sector, prf_path)
print(f"{'L px':>6s} {'direction':>9s} {'point/true':>11s} {'trailed/true':>13s}")
for L in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
    for ang in (0.0, np.pi / 4):
        nfr = 8; dt = 600 / 86400.0; t = np.arange(nfr) * dt
        x0_, y0_ = 24 + rng.uniform(-0.5, 0.5), 24 + rng.uniform(-0.5, 0.5)
        xs, ys = x0_ + L * np.arange(nfr) * np.cos(ang), y0_ + L * np.arange(nfr) * np.sin(ang)
        cubei = rng.normal(0, 2, (nfr, 128, 128)) + 100.0
        truth = np.zeros(nfr)
        for k_ in range(nfr):
            xc, yc = xs[k_], ys[k_]; xi_, yi_ = int(round(xc)), int(round(yc))
            K = max(int(np.ceil(L / 0.02)), 1) + 1; acc = 0.0
            for s_ in (np.linspace(-0.5, 0.5, K) if K > 1 else [0.0]):
                px, py = xc + s_ * L * np.cos(ang), yc + s_ * L * np.sin(ang)
                acc = acc + grid.images(np.array([X0 + px]), np.array([Y0 + py]), np.array([px - xi_]), np.array([py - yi_]), 15)[0]
            cubei[k_, yi_ - 15:yi_ + 16, xi_ - 15:xi_ + 16] += 10000.0 * acc / K
            truth[k_] = 10000.0 * grid.images(np.array([X0 + xc]), np.array([Y0 + yc]), np.array([xc - xi_]), np.array([yc - yi_]), 4)[0].sum()
        ephi = pd.DataFrame(dict(designation="z", frame=np.arange(nfr), mjd=t, x=xs, y=ys))
        fp = ap.forced_psf_photometry(cubei, ephi, sector, cam, ccd, X0, Y0, prf_path=prf_path).sort_values("frame")
        ft = ap.forced_psf_photometry(cubei, ephi, sector, cam, ccd, X0, Y0, prf_path=prf_path, exptime_s=600.0).sort_values("frame")
        rp, rt = fp.flux.to_numpy() / truth[fp.frame.to_numpy()], ft.flux.to_numpy() / truth[ft.frame.to_numpy()]
        print(f"{L:6.1f} {('x' if ang == 0 else 'diagonal'):>9s} {np.median(rp):11.3f} {np.median(rt):13.3f}   "
              f"(trailed scatter {np.std(rt):.3f}, trail_px {np.median(ft.trail_px):.2f}, {len(ft)} frames)")
        assert abs(np.median(rt) - 1) < 0.01, "trailed template does not recover the injected flux"
print("injection passed: trailed template within 1 % of the injected flux at every L")

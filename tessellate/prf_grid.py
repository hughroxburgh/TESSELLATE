"""
TESS PRF straight from the SPOC model files, for many sources at once.

The PRF package (TESS_PRF) builds one PRF for one detector position and places it with locate(); the
pipeline cached one per 100 px block. PRFGrid loads a CCD's 5x5 grid of model files once and gives
the pixel image of any number of sources, each blended bilinearly between the four grid PRFs around
its own detector position and interpolated between the 9x9 sub-pixel samples exactly as
TESS_PRF.locate does (same sample order, same border samples, normalised to 1), so for a single
source it reproduces locate() at that position.

Positions follow locate(): integers are pixel centres. stack_model() builds the image a stack of
whole-pixel stamps would show -- the mean over frames of each frame's PRF at its own sub-pixel
position -- by accumulating sub-pixel weights per detector-position group, so one model costs about
a millisecond however many frames go into it.
"""

import glob
import os

import numpy as np

NSAMP = 9            # PRF samples per pixel
PRF_SIZE = 13        # PRF model images are 13 x 13 pixels, centred on pixel 6
_SAMPLES = np.arange(-1 / 18, 19.1 / 18, 1 / 9)     # sub-pixel sample positions incl. border (11)
_GRIDS = {}


def _subsample_cube(prf):
    """117x117 interleaved PRF -> (11, 11, 13, 13) [row sample, col sample, row, col], with the
    border samples just beyond the pixel edges, as TESS_PRF.__init__ builds it."""
    out = np.zeros((11, 11, PRF_SIZE, PRF_SIZE))
    for i in range(NSAMP):
        for j in range(NSAMP):
            out[i + 1, j + 1] = prf[8 - i::NSAMP, 8 - j::NSAMP]
    for j in range(1, 10):
        out[j, 0] = np.append(out[j, -2, :, 1:], np.zeros((PRF_SIZE, 1)), axis=1)
        out[j, -1] = np.append(np.zeros((PRF_SIZE, 1)), out[j, 1, :, :-1], axis=1)
    for i in range(11):
        out[0, i] = np.append(out[-2, i, 1:, :], np.zeros((1, PRF_SIZE)), axis=0)
        out[-1, i] = np.append(np.zeros((1, PRF_SIZE)), out[1, i, :-1, :], axis=0)
    return out


def _sample_weights(frac):
    """Bracketing sub-pixel samples and the linear weight of the upper one, for fractional
    positions in [0, 1) (locate's frac = position + 0.5 mod 1)."""
    above = np.clip(np.searchsorted(_SAMPLES, frac, side='left'), 1, len(_SAMPLES) - 1)
    below = above - 1
    w = (frac - _SAMPLES[below]) / (_SAMPLES[above] - _SAMPLES[below])
    return below, above, w


class PRFGrid:
    """All PRF models of one camera/CCD, interpolated to any detector position."""

    def __init__(self, cam, ccd, sector, prf_path):
        prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
        files = [f for f in glob.glob(f'{prf_dir}/cam{int(cam)}_ccd{int(ccd)}/*.fits') if 'phot' not in f]
        if not files:
            raise FileNotFoundError(f'no PRF model files for Cam {cam} Ccd {ccd} in {prf_dir}')
        from astropy.io import fits
        self.rows = np.array(sorted({int(f[-17:-13]) for f in files}), dtype=float)
        self.cols = np.array(sorted({int(f[-9:-5]) for f in files}), dtype=float)
        self.cube = np.zeros((len(self.rows), len(self.cols), 11, 11, PRF_SIZE, PRF_SIZE))
        for f in files:
            r = np.searchsorted(self.rows, int(f[-17:-13]))
            c = np.searchsorted(self.cols, int(f[-9:-5]))
            self.cube[r, c] = _subsample_cube(fits.getdata(f).astype(float))

    def at(self, ccd_x, ccd_y):
        """The (11, 11, 13, 13) sub-sampled PRF at one detector position (bilinear over the grid,
        clamped to the grid's extent)."""
        def bracket(grid, v):
            k = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
            return k, float(np.clip((v - grid[k]) / (grid[k + 1] - grid[k]), 0, 1))
        r, ty = bracket(self.rows, ccd_y)
        c, tx = bracket(self.cols, ccd_x)
        return ((1 - ty) * (1 - tx) * self.cube[r, c] + (1 - ty) * tx * self.cube[r, c + 1]
                + ty * (1 - tx) * self.cube[r + 1, c] + ty * tx * self.cube[r + 1, c + 1])

    def stack_model(self, ccd_x, ccd_y, x, y, half, group_px=16):
        """Mean over sources of the PRF image each would give in a (2 half + 1)^2 stamp centred on
        pixel (half, half), for sources at stamp positions (half + x, half + y): x, y are offsets
        from the stamp centre (any size; the integer part moves the image). Sources share one
        grid-interpolated PRF per group_px x group_px block of the detector."""
        return StackModel(self, ccd_x, ccd_y, x, y, half, group_px)(0.0, 0.0)


class StackModel:
    """stack_model for one set of sources, re-evaluated at shifted positions without rebuilding
    anything: the detector-position groups and their grid-interpolated PRFs are made once."""

    def __init__(self, grid, ccd_x, ccd_y, x, y, half, group_px=16):
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.half, self.size = half, 2 * half + 1
        gx = np.floor(np.asarray(ccd_x) / group_px).astype(int)
        gy = np.floor(np.asarray(ccd_y) / group_px).astype(int)
        keys, self.group = np.unique(np.column_stack([gx, gy]), axis=0, return_inverse=True)
        self.group = self.group.ravel()
        # (groups, 121, 169): sub-pixel samples x pixels, for one matrix product per evaluation
        self.prfs = np.stack([grid.at((kx + 0.5) * group_px, (ky + 0.5) * group_px)
                              for kx, ky in keys]).reshape(len(keys), 121, PRF_SIZE * PRF_SIZE)
        self.n_groups = len(keys)

    def __call__(self, dx, dy):
        x, y = self.x + dx, self.y + dy
        sx = np.floor(x + 0.5).astype(int)                 # whole-pixel part, as locate's colint
        sy = np.floor(y + 0.5).astype(int)
        cb, _, cw = _sample_weights(x + 0.5 - sx)
        rb, _, rw = _sample_weights(y + 0.5 - sy)
        shifts, shift_id = np.unique(np.column_stack([sx, sy]), axis=0, return_inverse=True)
        key = self.group * len(shifts) + shift_id.ravel()
        w = np.zeros((self.n_groups * len(shifts), 11, 11))
        for dr, wr in ((0, 1 - rw), (1, rw)):
            for dc, wc in ((0, 1 - cw), (1, cw)):
                np.add.at(w, (key, rb + dr, cb + dc), wr * wc)
        w = w.reshape(self.n_groups, len(shifts), 121)
        n_src = w.sum(axis=2)                              # sources per (group, shift)
        imgs = np.einsum('gsk,gkp->gsp', w, self.prfs)      # (groups, shifts, 169)
        # normalised like locate: each source's interpolated 13x13 image sums to 1
        imgs *= (n_src / np.maximum(imgs.sum(axis=2), 1e-30))[..., None]
        imgs = imgs.sum(axis=0).reshape(len(shifts), PRF_SIZE, PRF_SIZE)
        model = np.zeros((self.size, self.size))
        pad = PRF_SIZE // 2
        for (ox, oy), img in zip(shifts, imgs):
            # paste the 13x13 PRF, centred on pixel 6, at stamp pixel (half + ox, half + oy)
            r0, c0 = self.half + oy - pad, self.half + ox - pad
            rs, cs = max(0, r0), max(0, c0)
            re, ce = min(self.size, r0 + PRF_SIZE), min(self.size, c0 + PRF_SIZE)
            if rs < re and cs < ce:
                model[rs:re, cs:ce] += img[rs - r0:re - r0, cs - c0:ce - c0]
        return model / len(x)


def get_grid(cam, ccd, sector, prf_path):
    key = (int(cam), int(ccd), sector >= 4, prf_path)
    if key not in _GRIDS:
        _GRIDS[key] = PRFGrid(cam, ccd, sector, prf_path)
    return _GRIDS[key]

"""
Draw the synthetic events made by ml_synthetic.py, laid out like the Navigator
plots used for manual sorting: the full-sector 3x3 light curve with a zoom on
the event, the brightest frame and the frame an hour later, and a strip of
frames around the peak. For judging how realistic the synthetic classes are.

Edit the CONFIG block, then:  python ml_plot_synthetic.py

Plots are written to OUT/{true class}_{injected kind}/S..C..C..C..O..E...png
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import KEY_COLS, _CutData, load_cut_events

# ---- CONFIG ----
SYNTH = f'{HERE}/synthetic'     # the OUT folder of ml_synthetic.py
SECTOR = 99
OUT = f'{SYNTH}/plots'
MAX_PER_KIND = 10               # events drawn per injected kind (None = all of them)
FRAME_BINS = [1]                # frame bins to draw (the synthetic data also has 3)
SEED = 0                        # which random events get picked
# ----------------


def sorted_keys(sort_dir):
    """Events that ml_synthetic.py put in the 'manually sorted' folders."""
    keys = set()
    for group in os.listdir(sort_dir):
        for name in os.listdir(os.path.join(sort_dir, group)):
            if name.endswith('.png'):
                s, rest = name[1:-4].split('C', 1)
                cam, ccd, rest = rest.split('C', 2)
                cut, rest = rest.split('O')
                objid, eventid = rest.split('E')
                keys.add((int(s), int(cam), int(ccd), int(cut), int(objid), int(eventid)))
    return keys


def plot_event(ev, cd, title, path):
    fb = int(ev.frame_bin)
    t, _, _ = cd.binned(fb)
    st = cd.stamp(int(ev.xint), int(ev.yint), 9, fb)               # (time, 19, 19) centred on the event
    core = st[:, 8:11, 8:11]
    lc = np.nansum(core, axis=(1, 2))
    lc[~np.isfinite(core).any(axis=(1, 2))] = np.nan
    fs, fe, fm = int(ev.frame_start), int(ev.frame_end), int(ev.frame_max)
    tt = t - t[0]
    cad = np.nanmedian(np.diff(tt))

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.75], hspace=0.35)

    # -- Full light curve, broken at data gaps, with the event shaded -- #
    ax = fig.add_subplot(gs[:2, :3])
    segments = np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt) > 5 * cad) + 1)
    for seg in segments:
        ax.plot(tt[seg], lc[seg], 'k', lw=0.6, alpha=0.8)
    ax.axvspan(tt[fs] - cad / 2, tt[fe] + cad / 2, color='C1', alpha=0.4)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo))                            # room for the zoom, as in Plot_LC_Frame
    ax.set_xlim(tt.min(), tt.max())
    ax.set_xlabel(f'Time (MJD - {t[0]:.3f})')
    ax.set_ylabel('Counts (3x3 sum)')

    # -- Zoom on the event: 3 durations either side in time, as in Plot_LC_Frame -- #
    ins = ax.inset_axes([0.1, 0.55, 0.86, 0.43])
    dur = max(fe - fs + 1, 4)
    x0, x1 = tt[fs] - 3 * dur * cad, tt[fe] + 3 * dur * cad
    for seg in segments:
        ins.plot(tt[seg], lc[seg], 'k.-', lw=0.8, ms=3)
    ins.axvspan(tt[fs] - cad / 2, tt[fe] + cad / 2, color='C1', alpha=0.4)
    ins.set_xlim(x0, x1)
    shown = (tt >= x0) & (tt <= x1) & np.isfinite(lc)
    if shown.any():
        pad = 0.05 * (np.nanmax(lc[shown]) - np.nanmin(lc[shown]) + 1)
        ins.set_ylim(np.nanmin(lc[shown]) - pad, np.nanmax(lc[shown]) + pad)
    for spine in ins.spines.values():
        spine.set_color('r')
        spine.set_linewidth(2)

    # -- Brightest frame and an hour later -- #
    img = st[fm]
    vmin = np.nanpercentile(img, 16)
    vmax = np.nanpercentile(img[8:11, 8:11], 90)
    if vmin >= vmax:
        vmin = vmax - 5
    later = min(int(np.searchsorted(tt, tt[fm] + 1 / 24)), len(tt) - 1)
    cx, cy = ev.xcentroid - (int(ev.xint) - 9), ev.ycentroid - (int(ev.yint) - 9)
    for row, (frame, name) in enumerate([(fm, 'Brightest image'), (later, '1 hour later')]):
        a = fig.add_subplot(gs[row, 3:])
        a.imshow(st[frame], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        a.add_patch(Rectangle((6.5, 6.5), 5, 5, lw=2, ec='r', fc='none'))
        if row == 0:
            a.scatter(cx, cy, color='r', marker='x', s=50, lw=2)
        a.set_title(name)
        a.axis('off')

    # -- Strip of frames around the peak -- #
    strip = st[:, 4:15, 4:15]
    svmin = np.nanpercentile(strip[fm, 4:7, 4:7], 10)
    svmax = np.nanpercentile(strip[fm, 4:7, 4:7], 90)
    for k, frame in enumerate(range(fm - 2, fm + 3)):
        a = fig.add_subplot(gs[2, k])
        if 0 <= frame < len(tt):
            a.imshow(strip[frame], origin='lower', cmap='gray', vmin=svmin, vmax=svmax)
        a.set_title('Brightest' if frame == fm else f'Frame {frame}', fontsize=9)
        a.axis('off')

    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    truth = pd.read_csv(f'{SYNTH}/truth.csv')
    truth = truth[(truth.sector == SECTOR) & truth.frame_bin.isin(FRAME_BINS)]
    sorted_set = sorted_keys(f'{SYNTH}/sorted')

    rng = np.random.default_rng(SEED)
    picks = []
    for _, grp in truth.groupby('kind'):
        n = len(grp) if MAX_PER_KIND is None else min(MAX_PER_KIND, len(grp))
        picks.append(grp.iloc[rng.choice(len(grp), n, replace=False)])
    picks = pd.concat(picks)

    count = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for (cam, ccd, cut), grp in picks.groupby(['camera', 'ccd', 'cut']):
            data_path = f'{SYNTH}/TESSdata'
            cd = _CutData(data_path, SECTOR, cam, ccd, cut, frame_stats=False)
            events = load_cut_events(data_path, SECTOR, cam, ccd, cut).merge(grp[KEY_COLS + ['true_class', 'kind']],
                                                                            on=KEY_COLS)
            for ev in events.itertuples():
                key = tuple(int(getattr(ev, k)) for k in KEY_COLS)
                label = 'in manual sort' if key in sorted_set else 'not sorted'
                tag = ev.classification if ev.classification != '-' else 'none'
                title = (f'{ev.true_class} ({ev.kind})   |   pipeline tag: {tag}   |   {label}   |   '
                         f'frame_bin {ev.frame_bin}, {ev.frame_duration} frames')
                folder = f'{OUT}/{ev.true_class}_{ev.kind}'
                os.makedirs(folder, exist_ok=True)
                plot_event(ev, cd, title, f'{folder}/S{key[0]}C{key[1]}C{key[2]}C{key[3]}O{key[4]}E{key[5]}.png')
                count += 1
    print(f'{count} plots written to {OUT}')


if __name__ == '__main__':
    main()

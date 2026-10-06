"""
Score the stage-1 model against a sort of a fresh sector's stratified random sample (ml_sector_sample.py on
the cluster, sorted with manual_sort), weighted back to every frame-bin-1 event of the sector. Runs locally.

Each sorted event stands for N_stratum / n_sorted events of its stratum (re-weighted over the events actually
sorted, so unplotted or skipped events don't bias it). Unsure is left out. Uncertainties: bootstrap within
each stratum (BOOT resamples), as 16-84% ranges.

Reported for lc_sig_max >= 5 (the periodic, sig_5_10 and sig_10_plus strata) and for below_5 separately:
  - what the events really are (weighted fractions)
  - the model's confusion (raw counts and weighted fractions)
  - throwing away p(Junk) + p(CosmicRay) >= 0.5: real events kept, artefacts removed, purity of what is kept
  - every event the model got wrong

Edit the CONFIG block, then:  python ml_eval_sector_sort.py
"""

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
INFO = f'{HERE}/ml_data/v5/sort_sample_info.csv'                 # ml_sector_sample.py's sort_sample_info.csv
SORT_DIR = f'{HERE}/ml_data/v5/S54_sort/sort_sort_sample'         # manual_sort folder (one subfolder per label)
OUT_FILE = f'{HERE}/ml_eval_sector_sort_S54.txt'
OOF = None                  # None = score the model that made INFO's predictions. Or an oof_predictions.csv from
                            # ml_train_eval.py: score that model's held-out predictions instead (for a model trained
                            # with this sector's labels; events without one are dropped)
CLASS_MERGE = {'Systematic': 'Junk', 'Blend': 'Junk', 'Noise': 'Junk'}
LEAVE_OUT = ['Unsure']
BOOT = 2000
SEED = 0
# ----------------

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
ARTEFACTS = ['Junk', 'CosmicRay']
CLASSES = ['Junk', 'CosmicRay', 'Asteroid', 'Flare', 'Variable']


def load_labels():
    parts = [pd.read_csv(f).assign(label=os.path.basename(os.path.dirname(f)))
             for f in glob.glob(f'{SORT_DIR}/*/events.csv')]
    lab = pd.concat(parts, ignore_index=True)
    lab['label'] = lab.label.replace(CLASS_MERGE)
    return lab[~lab.label.isin(LEAVE_OUT)].drop_duplicates(KEY_COLS)


def rates(d):
    """Weighted: real kept, artefacts removed, purity of kept, accuracy (5 classes)."""
    w = d.weight.to_numpy()
    real = ~d.label.isin(ARTEFACTS).to_numpy()
    kept = (d[[f'p_{c}' for c in ARTEFACTS]].sum(axis=1) < 0.5).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'real kept': np.sum(w * (real & kept)) / np.sum(w * real),
                'artefacts removed': np.sum(w * (~real & ~kept)) / np.sum(w * ~real),
                'purity of kept': np.sum(w * (real & kept)) / np.sum(w * kept),
                'accuracy (5 classes)': np.sum(w * (d.pred_class == d.label).to_numpy()) / np.sum(w)}


def boot(d, rng):
    """16-84% bootstrap ranges of rates(), resampling events within each stratum."""
    groups = [g for _, g in d.groupby('stratum')]
    draws = []
    for _ in range(BOOT):
        s = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        draws.append(rates(s))
    return pd.DataFrame(draws).quantile([0.16, 0.84])


def section(d, title, rng):
    lines = ['=' * 78, f'{title}: {len(d)} sorted events', '=' * 78]
    if not len(d):
        return lines
    frac = d.groupby('label').weight.sum() / d.weight.sum()
    lines += ['What they really are (weighted fractions of the population):',
              frac.reindex(CLASSES).fillna(0).round(3).to_string(), '']
    lines += ['Model confusion, raw counts (rows = sort, columns = model):',
              pd.crosstab(d.label, d.pred_class).to_string(), '']
    wt = pd.crosstab(d.label, d.pred_class, values=d.weight, aggfunc='sum').fillna(0) / d.weight.sum()
    lines += ['Model confusion, weighted fractions of the population:', wt.round(3).to_string(), '']
    r, b = rates(d), boot(d, rng)
    lines += ['Throwing away p(Junk) + p(CosmicRay) >= 0.5 (16-84% bootstrap range):']
    for k, v in r.items():
        lines.append(f'  {k:22s} {v:.3f}   ({b[k].iloc[0]:.3f} - {b[k].iloc[1]:.3f})')
    lines += ['', 'By stratum (raw):']
    for name, g in d.groupby('stratum'):
        lines.append(f'  {name:12s} n={len(g):3d}  right {np.mean(g.pred_class == g.label):.2f}  '
                     f'(stands for {int(g.n_stratum.iloc[0]):,} events)')
    wrong = d[d.pred_class != d.label].sort_values(['label', 'pred_class'])
    cols = KEY_COLS + ['stratum', 'label', 'pred_class'] + [f'p_{c}' for c in CLASSES] + \
        ['lc_sig_max', 'duration_hr', 'ls_power', 'n_comparable50', 'flux_sign', 'classification']
    lines += ['', f'Model wrong ({len(wrong)}):', wrong[[c for c in cols if c in wrong]].round(2).to_string(index=False), '']
    return lines


def main():
    rng = np.random.default_rng(SEED)
    info = pd.read_csv(INFO)
    lab = load_labels()
    if OOF:
        oof = pd.read_csv(OOF).drop_duplicates(KEY_COLS)
        p_cols = [f'p_{c}' for c in CLASSES]
        oof['pred_class'] = np.array(CLASSES)[oof[p_cols].to_numpy().argmax(axis=1)]
        info = info.drop(columns=p_cols + ['pred_class']).merge(oof[KEY_COLS + p_cols + ['pred_class']], on=KEY_COLS)
    d = info.merge(lab[KEY_COLS + ['label']], on=KEY_COLS, how='inner')
    n_sorted = d.groupby('stratum').size()
    d['weight'] = d.n_stratum / d.stratum.map(n_sorted)      # re-weight over the events actually sorted
    lines = [f'Sort {SORT_DIR} vs {INFO}' + (f', held-out predictions from {OOF}' if OOF else ''),
             f'{len(info)} picked, {len(lab)} labelled (excl. {LEAVE_OUT}), {len(d)} matched; '
             f'sorted per stratum: {n_sorted.to_dict()}', '']
    lines += section(d[d.stratum != 'below_5'], 'lc_sig_max >= 5', rng)
    lines += section(d[d.stratum == 'below_5'], 'lc_sig_max < 5', rng)
    text = '\n'.join(lines)
    with open(OUT_FILE, 'w') as f:
        f.write(text + '\n')
    print(text)
    print(f'\n-> {OUT_FILE}')


if __name__ == '__main__':
    main()

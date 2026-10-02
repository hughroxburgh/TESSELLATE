"""
Score a stage-1 model on the random sort, weighted back to every frame-bin-1 event of the pool's sectors
(runs locally).

The sort sample was drawn by strata (ml_pick_sort_sample.py STRATA) from the random pool (ml_random_pool.py).
Each sorted event stands for 1 / (p_pool * p_sorted) events, where p_sorted = sorted / pool events in its
stratum, counted within SECTORS -- so sectors that weren't sorted, and events left unsorted, drop out cleanly
(assuming the unsorted ones are a random subset of their stratum).

Prints, for the population above SIG_FLOOR (and per stratum):
  - what the events really are (weighted class fractions)
  - the model's confusion against the sort (raw counts, and weighted)
  - artefact vs astrophysical: real events kept, artefacts removed, purity of what's kept
The model is scored fresh from its saved file on the pool's features.

Edit the CONFIG block, then:  python ml_eval_random_sort.py
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import ARTEFACT_CLASSES, KEY_COLS, EventClassifier

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
MODEL = f'{HERE}/ml_data/event_classifier_stage1_fb1.joblib'
OOF = f'{HERE}/ml_eval_stage1_fb1_random/oof_predictions.csv'                          # oof_predictions.csv from ml_train_eval.py: use its held-out probabilities
                                    # instead of MODEL (for a model trained on the random sort itself)
LABELS = f'{HERE}/ml_data/manual_labels_random.csv'               # ml_build_labels.py output for the random sort
POOL_FEATURES = f'{HERE}/ml_data/S27-55_training_features_pool.csv.gz'
POOL_CSV = f'{HERE}/ml_data/random_pool.csv'
SECTORS = list(range(27, 40))       # sectors the sort covers (S55's sample wasn't plotted)
SIG_FLOOR = 5                       # the population scored: lc_sig_max >= this
CLASS_MERGE = {'Systematic': 'Junk', 'Blend': 'Junk', 'Noise': 'Junk'}
P_ARTEFACT_CUT = 0.5                # an event is thrown away when p(artefact) >= this
OUT_FILE = f'{HERE}/ml_eval_random_sort_after.txt'
# ----------------


def strata_of(features):
    """Each pool event's stratum, by the same rules ml_pick_sort_sample.py used."""
    spec = importlib.util.spec_from_file_location('pick', f'{HERE}/ml_pick_sort_sample.py')
    pick = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pick)
    stratum = np.full(len(features), '', dtype=object)
    for name, rule, _ in pick.STRATA:
        hit = features.eval(rule).fillna(False).to_numpy(bool) & (stratum == '')
        stratum[hit] = name
    return stratum


def weighted_table(rows, cols, w):
    return pd.crosstab(rows, cols, values=w, aggfunc='sum').fillna(0)


def main():
    pool = pd.read_csv(POOL_CSV)
    feats = pd.read_csv(POOL_FEATURES, low_memory=False)
    feats = feats[feats.frame_bin == 1].merge(pool[KEY_COLS + ['p_pool']], on=KEY_COLS)
    feats = feats[feats.sector.isin(SECTORS)].reset_index(drop=True)
    feats['stratum'] = strata_of(feats)

    labels = pd.read_csv(LABELS)
    labels['label'] = labels['label'].replace(CLASS_MERGE)
    df = feats.merge(labels[KEY_COLS + ['label']], on=KEY_COLS, how='left')

    n_pool = df.groupby('stratum').size()
    n_sorted = df[df.label.notna()].groupby('stratum').size()
    p_sorted = (n_sorted / n_pool).reindex(n_pool.index).fillna(0)
    s = df[df.label.notna()].copy()
    s['weight'] = 1 / (s['p_pool'] * s['stratum'].map(p_sorted))

    if OOF:
        oof = pd.read_csv(OOF)
        p_cols = [c for c in oof if c.startswith('p_') and c[2:] in set(oof.label)]
        s = s.merge(oof[KEY_COLS + p_cols], on=KEY_COLS, how='inner')
        classes = [c[2:] for c in p_cols]
        s['pred'] = np.array(classes)[s[p_cols].to_numpy().argmax(axis=1)]
        s['p_art'] = s[[f'p_{c}' for c in classes if c in ARTEFACT_CLASSES]].sum(axis=1).to_numpy()
    else:
        clf = EventClassifier.load(MODEL)
        pred = clf.predict(s)
        s['pred'] = pred['pred_class'].to_numpy()
        s['p_art'] = pred[[f'p_{c}' for c in clf.classes_ if c in ARTEFACT_CLASSES]].sum(axis=1).to_numpy()
    s['is_art'] = s['label'].isin(ARTEFACT_CLASSES)
    s['thrown'] = s['p_art'] >= P_ARTEFACT_CUT

    out = [f'Random sort vs {os.path.basename(OOF or MODEL)}; sectors {SECTORS[0]}-{SECTORS[-1]}; '
           f'weights scale to all frame-bin-1 events there', '']
    out += ['Sorted events by stratum (pool events in brackets):']
    out += [f'  {k:12s} {int(n_sorted.get(k, 0)):4d}  ({int(n_pool[k])})' for k in n_pool.index if k]
    pop = s[s['tab_lc_sig_max'] >= SIG_FLOOR]
    for name, d in [(f'lc_sig_max >= {SIG_FLOOR}', pop)] + [(f'stratum {k}', g) for k, g in s.groupby('stratum')]:
        w = d['weight']
        frac = (w.groupby(d['label']).sum() / w.sum()).round(3)
        real = ~d['is_art']
        kept_real = (w[real & ~d['thrown']].sum() / max(w[real].sum(), 1e-12))
        removed_art = (w[d['is_art'] & d['thrown']].sum() / max(w[d['is_art']].sum(), 1e-12))
        purity = (w[real & ~d['thrown']].sum() / max(w[~d['thrown']].sum(), 1e-12))
        out += ['', '=' * 70, f'{name}: {len(d)} sorted events', '=' * 70,
                'What they really are (weighted fractions):', frac.to_string(),
                '', 'Model confusion, raw counts (rows = sort, columns = model):',
                pd.crosstab(d['label'], d['pred']).to_string(),
                '', 'Model confusion, weighted fractions of the population:',
                (weighted_table(d['label'], d['pred'], w) / w.sum()).round(3).to_string(),
                '', f'Throwing away p(artefact) >= {P_ARTEFACT_CUT}:',
                f'  real (non-artefact) events kept   {kept_real:.3f}',
                f'  artefacts removed                 {removed_art:.3f}',
                f'  purity of what is kept            {purity:.3f}']
    text = '\n'.join(out) + '\n'
    with open(OUT_FILE, 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()

"""
Pick the events to sort by eye from the random pool, and score them with the stage-1 model (runs locally).

Every pool event goes into the first stratum in STRATA whose rule it meets (a pandas query on the pool's
feature table; events meeting none are not sampled), and each stratum's n events are drawn at random from it.
The strata are rules, not model scores: outside the sorted domain (lc_sig_max >= 10) the model can't be trusted
yet -- on the random pool it called 68% of events Asteroid, mostly faint non-moving blips. Every picked event carries its weight,
1 / (p_pool * p_stratum), so counts from the sort can be scaled back to all frame-bin-1 events of the sectors.

Outputs in OUT_DIR:
  sort_sample.csv       the events to plot and sort (event columns only -- nothing that hints at the answer)
  sort_sample_info.csv  the same events with stratum, weights and the model's probabilities (for afterwards)

Edit the CONFIG block, then:  python ml_pick_sort_sample.py
Then plot the events on the cluster (ml_plot_sort_sample.py) and sort them with tools.manual_sort.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import ARTEFACT_CLASSES, KEY_COLS, EventClassifier

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
POOL_FEATURES = f'{HERE}/ml_data/S27-55_training_features_pool.csv.gz'   # ml_collect_features.py output for the pool
POOL_CSV = f'{HERE}/ml_data/random_pool.csv'                    # ml_random_pool.py output (has p_pool)
MODEL = f'{HERE}/ml_data/event_classifier_stage1_fb1.joblib'
FRAME_BINS = [1]

STRATA = [                  # (name, rule, n): first matching rule wins
    ('periodic',    'tab_lc_sig_max >= 5 and ctx_ls_power >= 0.3', 250),   # weak/long variables, flares on them
    ('sig_5_10',    'tab_lc_sig_max >= 5 and tab_lc_sig_max < 10', 400),   # below the sorted domain
    ('sig_10_plus', 'tab_lc_sig_max >= 10', 250),                          # the sorted domain: checks the model
    ('below_5',     'tab_lc_sig_max < 5', 100),                            # is it all noise down there?
]
SEED = 29

OUT_DIR = f'{HERE}/ml_data/sort_sample'
# ----------------


def main():
    pool = pd.read_csv(POOL_CSV)
    features = pd.read_csv(POOL_FEATURES, low_memory=False)
    features = features[features.frame_bin.isin(FRAME_BINS)]
    features = features.merge(pool[KEY_COLS + ['p_pool']], on=KEY_COLS, how='inner').reset_index(drop=True)
    print(f'{len(features)} of {len(pool)} pool events have features')

    clf = EventClassifier.load(MODEL)
    pred = clf.predict(features)
    p_cols = [f'p_{c}' for c in clf.classes_]
    p_art = pred[[f'p_{c}' for c in clf.classes_ if c in ARTEFACT_CLASSES]].sum(axis=1).to_numpy()
    ls = features['ctx_ls_power'].to_numpy()

    stratum = np.full(len(features), '', dtype=object)
    for name, rule, _ in STRATA:
        hit = features.eval(rule).fillna(False).to_numpy(bool) & (stratum == '')
        stratum[hit] = name
    info = features[KEY_COLS + ['frame_bin', 'classification', 'p_pool']].copy()
    info['stratum'] = stratum
    info['p_artefact'] = p_art
    info['ctx_ls_power'] = ls
    info['lc_sig_max'] = features['tab_lc_sig_max'].to_numpy()
    info[p_cols] = pred[p_cols].to_numpy()
    info['pred_class'] = pred['pred_class'].to_numpy()

    rng = np.random.default_rng(SEED)
    picked = []
    print('\nstratum          pool   picked')
    for name, _, n in STRATA:
        idx = np.flatnonzero(stratum == name)
        k = min(n, len(idx))
        take = rng.choice(idx, k, replace=False) if k else np.array([], int)
        picked.append(info.iloc[take].assign(p_stratum=k / max(len(idx), 1)))
        print(f'{name:15s} {len(idx):6d}   {k:6d}')
    sample = pd.concat(picked).sample(frac=1, random_state=SEED)   # shuffled: strata mixed when sorting
    sample['weight'] = 1 / (sample['p_pool'] * sample['p_stratum'])

    os.makedirs(OUT_DIR, exist_ok=True)
    sample[KEY_COLS + ['frame_bin']].to_csv(f'{OUT_DIR}/sort_sample.csv', index=False)
    sample.to_csv(f'{OUT_DIR}/sort_sample_info.csv', index=False)
    print(f'\n{len(sample)} events -> {OUT_DIR}/sort_sample.csv')
    print('\nModel prediction by stratum (picked events):')
    print(pd.crosstab(sample['stratum'], sample['pred_class']).to_string())


if __name__ == '__main__':
    main()

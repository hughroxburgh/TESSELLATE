"""
How well can we flag a class the model has never seen? (runs locally)

For each class in LEAVE_OUT, pretend it doesn't exist: with whole sectors held out in turn, fit the classifier
and the novelty scores on the OTHER sectors' labelled events of the remaining classes, then score the held-out
sector's events -- the left-out class plays a new, never-seen kind of object; the rest are ordinary events.

Novelty signals compared (higher = more unusual):
  low_conf   1 - the classifier's top probability (only says "between known classes")
  iforest    isolation forest on the classifier's features (rank-scaled, NaN -> median)
  knn        mean distance to the K nearest labelled training events, same scaling
  knn_class  the same, but only to training events of the class the classifier predicted (an unseen flare
             filed as CosmicRay may still be far from real cosmic rays)
  combined   the larger of the knn_class and low_conf percentile ranks

Reported per left-out class: ROC AUC (unseen vs ordinary), and the share of the unseen class flagged when the
threshold is set so FLAG_RATE of ordinary events are flagged (the review-pile cost), plus what the classifier
filed the unseen class as.

Edit the CONFIG block, then:  python ml_novelty_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import HOST_FEATURES, KEY_COLS, LEAKY_FEATURES, _feature_columns

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
FEATURES = f'{HERE}/ml_data/v4/S27-55_training_features_v4.csv.gz'
LABELS = [f'{HERE}/ml_data/v4/manual_labels_all.csv', f'{HERE}/ml_data/v4/manual_labels_round3.csv']
CLASS_MERGE = {'Systematic': 'Junk', 'Blend': 'Junk', 'Noise': 'Junk'}
DROP_LABELS = ['Interesting']
EXCLUDE = HOST_FEATURES + LEAKY_FEATURES + ('tab_abs_gal_b',)
LEAVE_OUT = ['Flare', 'Variable', 'Asteroid']
FLAG_RATE = 0.05            # share of ordinary (known-class) events we accept sending to review
K = 10                      # neighbours for the knn score
SEED = 0
OUT_FILE = f'{HERE}/ml_eval_novelty.txt'
# ----------------


def load():
    labels = pd.concat([pd.read_csv(f) for f in LABELS], ignore_index=True)
    labels['label'] = labels['label'].replace(CLASS_MERGE)
    labels = labels[~labels.label.isin(DROP_LABELS)].drop_duplicates(KEY_COLS + ['label'])
    labels = labels[~labels.duplicated(KEY_COLS, keep=False)]
    feats = pd.read_csv(FEATURES, low_memory=False)
    feats = feats[feats.frame_bin == 1].drop_duplicates(KEY_COLS)
    df = feats.merge(labels[KEY_COLS + ['label']], on=KEY_COLS)
    cols = _feature_columns(df.columns, exclude=EXCLUDE)
    return df.reset_index(drop=True), cols


def class_weights(y):
    counts = pd.Series(y).value_counts()
    return pd.Series(y).map(lambda c: (len(y) / (len(counts) * counts[c])) ** 0.5).to_numpy()   # half-balanced


def scores_for_fold(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import QuantileTransformer

    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=SEED)
    clf.fit(Xtr, ytr, sample_weight=class_weights(ytr))
    proba = clf.predict_proba(Xte)
    pred = clf.classes_[proba.argmax(axis=1)]

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda X: np.where(np.isfinite(X), X, med)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(Xtr)), random_state=SEED)
    Ztr = qt.fit_transform(fill(Xtr))
    Zte = qt.transform(fill(Xte))

    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(Ztr)
    nn = NearestNeighbors(n_neighbors=K).fit(Ztr)
    dist = nn.kneighbors(Zte)[0].mean(axis=1)
    dist_class = np.full(len(Zte), np.nan)
    for c in clf.classes_:
        tr_c, te_c = ytr == c, pred == c
        if te_c.any() and tr_c.sum() > K:
            dist_class[te_c] = NearestNeighbors(n_neighbors=K).fit(Ztr[tr_c]).kneighbors(Zte[te_c])[0].mean(axis=1)
    return dict(low_conf=1 - proba.max(axis=1), iforest=-iso.score_samples(Zte), knn=dist,
                knn_class=dist_class), pred


def evaluate(df, cols, left_out):
    from sklearn.metrics import roc_auc_score

    X = df[cols].to_numpy(float)
    y = df['label'].to_numpy()
    out = []
    for sector in sorted(df.sector.unique()):
        te = (df.sector == sector).to_numpy()
        tr = ~te & (y != left_out)
        if tr.sum() < 100 or not te.any():
            continue
        s, pred = scores_for_fold(X[tr], y[tr], X[te])
        out.append(pd.DataFrame({**s, 'pred': pred, 'label': y[te], 'sector': sector}))
    res = pd.concat(out, ignore_index=True)
    # combined: rank-average-free version -- the larger of the two percentile ranks
    res['combined'] = np.maximum(res['knn_class'].rank(pct=True), res['low_conf'].rank(pct=True))

    unseen = (res.label == left_out).to_numpy()
    rows = []
    for m in ['low_conf', 'iforest', 'knn', 'knn_class', 'combined']:
        thr = np.quantile(res.loc[~unseen, m], 1 - FLAG_RATE)
        rows.append({'signal': m, 'AUC': roc_auc_score(unseen, res[m]),
                     f'unseen flagged @ {FLAG_RATE:.0%} ordinary': float((res.loc[unseen, m] > thr).mean())})
    filed = res.loc[unseen, 'pred'].value_counts(normalize=True).round(3).to_dict()
    return pd.DataFrame(rows), int(unseen.sum()), filed


def main():
    df, cols = load()
    print(f'{len(df)} labelled frame-bin-1 events, {len(cols)} features; classes {df.label.value_counts().to_dict()}')
    text = [f'Novelty test: each class left out in turn; whole sectors held out; flag threshold set so '
            f'{FLAG_RATE:.0%} of ordinary events are flagged.', '']
    for left_out in LEAVE_OUT:
        table, n, filed = evaluate(df, cols, left_out)
        block = [f'== {left_out} unseen ({n} events) ==', table.round(3).to_string(index=False),
                 f'classifier filed them as: {filed}', '']
        print('\n'.join(block), flush=True)
        text += block
    with open(OUT_FILE, 'w') as f:
        f.write('\n'.join(text) + '\n')
    print(f'-> {OUT_FILE}')


if __name__ == '__main__':
    main()

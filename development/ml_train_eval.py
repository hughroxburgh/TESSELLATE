"""
Train and evaluate the tessellate event classifier from extracted features
(ml_extract_features.py) and manual_sort.py labels.

Edit the CONFIG block, then:  python ml_train_eval.py

Every metric is out-of-fold (folds are whole cuts) and computed on manual
labels only; pipeline tags and crossbin-inherited labels are used for
training (down-weighted), never for scoring. Written to OUT_DIR:
  report.txt            per-class and artefact-vs-astrophysical metrics
  oof_predictions.csv   cross-validated predictions for every labelled event
  confusion.png  reliability.png  p_astro_by_class.png  importance.png
  ablation.csv          the same metrics for subsets of feature groups (ABLATION)
  review_queue.csv      unlabelled events ranked for manual sorting (REVIEW)
  truth_check.txt       accuracy on never-labelled events against TRUTH (synthetic data)
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import (ARTEFACT_CLASSES, FEATURE_GROUPS, EventClassifier, attach_labels, evaluate,
                                      load_manual_labels, rank_for_review)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
FEATURES = [f'{HERE}/S55_features.csv.gz']   # one or more files from ml_extract_features.py
SORT_DIR = f'{HERE}/images'                  # manual_sort.py output: one folder per group
OUT_DIR = f'{HERE}/ml_eval'

PIPELINE_WEIGHT = 0.3       # weight of the pipeline's Junk/CosmicRay/Asteroid tags (0 = ignore them)
CROSSBIN_WEIGHT = 0.5       # weight of labels inherited by other-frame-bin detections of a sorted event (0 = off)
GROUPS = FEATURE_GROUPS     # feature groups to use: 'tab', 'lc', 'shape', 'ctx', 'pix', 'xm'
CLASS_BALANCE = 0.5         # 0 = natural class frequencies, 1 = fully balanced
N_SPLITS = 5

ABLATION = True             # also score subsets of feature groups (slower)
IMPORTANCE = True           # permutation importance per feature and per group
SAVE_MODEL = f'{HERE}/event_classifier.joblib'   # None = don't save
REVIEW = 500                # write the N most useful unlabelled events to sort next (None = skip)
TRUTH = None                # csv of true classes, for synthetic data only
# ----------------

ABLATIONS = [('tab',), ('tab', 'xm'), ('tab', 'lc', 'shape'), ('tab', 'lc', 'shape', 'ctx'),
             ('tab', 'lc', 'shape', 'ctx', 'pix'), ('lc', 'shape', 'ctx', 'pix'), FEATURE_GROUPS]


def format_report(res, title):
    lines = [title, '=' * len(title),
             f"n = {res['n']}   baseline purity (astrophysical fraction of these events) = {res['baseline_purity']:.3f}",
             '', 'Per class:', res['per_class'].round(3).to_string(),
             '', 'Confusion (rows = label, columns = prediction):', res['confusion'].to_string(),
             '', 'Artefact vs astrophysical:']
    lines += [f'  {k:36s} {v:.4f}' for k, v in res['coarse'].items()]
    lines += ['', 'Calibration:'] + [f'  {k:36s} {v:.4f}' for k, v in res['calibration'].items()]
    return '\n'.join(lines) + '\n'


def summary_row(res):
    row = {'n': res['n'], 'macro_f1': res['per_class'].loc['macro avg', 'f1-score']}
    row.update({k: res['coarse'].get(k, np.nan) for k in ['roc_auc', 'average_precision',
                                                           'recall95_artefacts_removed', 'recall95_purity']})
    row.update({k: res['calibration'][k] for k in ['log_loss', 'ece']})
    return row


def plot_confusion(cm, path):
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(1.2 * len(cm) + 2, 1.1 * len(cm) + 1.5))
    ax.imshow(norm.to_numpy(), cmap='Blues', vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f'{cm.iat[i, j]}\n{norm.iat[i, j]:.2f}', ha='center', va='center',
                    color='white' if norm.iat[i, j] > 0.5 else 'black', fontsize=8)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(cm)), cm.index)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Manual label')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(oof_raw, oof_cal, classes, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    for df, name in [(oof_raw, 'uncalibrated'), (oof_cal, 'calibrated')]:
        df = df[df.label_source == 'manual']
        P = df[[f'p_{c}' for c in classes]].to_numpy()
        conf = P.max(axis=1)
        correct = np.array(classes)[P.argmax(axis=1)] == df.label.to_numpy()
        which = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        centres = [conf[which == b].mean() for b in range(10) if np.sum(which == b) >= 5]
        acc = [correct[which == b].mean() for b in range(10) if np.sum(which == b) >= 5]
        ax.plot(centres, acc, 'o-', label=name)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('Predicted probability of the top class')
    ax.set_ylabel('Fraction correct')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_p_astro(oof, classes, path):
    df = oof[oof.label_source == 'manual']
    astro = [c for c in classes if c not in ARTEFACT_CLASSES]
    p = df[[f'p_{c}' for c in astro]].sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, grp in df.groupby('label'):
        ax.hist(p[grp.index], bins=np.linspace(0, 1, 26), histtype='step', lw=2, label=f'{label} ({len(grp)})')
    ax.set_xlabel('p(astrophysical), out-of-fold')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp, path, top=25):
    feats = imp[imp.kind == 'feature'].sort_values('mean', ascending=False).head(top)[::-1]
    groups = imp[imp.kind == 'group'].sort_values('mean', ascending=False)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.28 * top + 1.5), gridspec_kw={'width_ratios': [3, 1.4]})
    ax[0].barh(feats.name, feats['mean'], xerr=feats['std'])
    ax[0].set_xlabel('Increase in log-loss when permuted')
    ax[0].set_title('Top features')
    ax[1].barh(groups.name, groups['mean'], xerr=groups['std'])
    ax[1].set_title('Whole feature groups')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def truth_check(predictions, labels, truth, path):
    """Accuracy on events the model never saw a label for, against known true classes."""
    unlabelled = predictions[labels.label.reindex(predictions.index).isna()]
    check = unlabelled.merge(truth, on=['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid'], suffixes=('', '_truth'))
    lines = [f'{len(check)} events the model never saw a label for (unsorted, or other frame bins '
             f'without a sorted sibling)',
             f"fine-class accuracy: {np.mean(check.pred_class == check.true_class):.3f}",
             f"coarse accuracy: {np.mean((check.pred_coarse == 'Artefact') == check.true_class.isin(ARTEFACT_CLASSES)):.3f}",
             '', pd.crosstab(check.true_class, check.pred_class).to_string(),
             '', 'accuracy by frame_bin:',
             check.groupby('frame_bin').apply(lambda g: np.mean(g.pred_class == g.true_class)).round(3).to_string()]
    if 'kind' in check:
        lines += ['', 'accuracy by injected kind:',
                  check.groupby('kind').apply(lambda g: np.mean(g.pred_class == g.true_class)).round(3).to_string()]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


def run(features_files, sort_dir, out_dir, pipeline_weight=0.3, crossbin_weight=0.5, groups=FEATURE_GROUPS,
        class_balance=0.5, n_splits=5, ablation=False, importance=False, save_model=None, review=None, truth=None):
    os.makedirs(out_dir, exist_ok=True)
    features = pd.concat([pd.read_csv(f) for f in features_files], ignore_index=True)
    manual = load_manual_labels(sort_dir) if sort_dir else None
    labels = attach_labels(features, manual, pipeline_weight=pipeline_weight, crossbin_weight=crossbin_weight)

    clf = EventClassifier(feature_groups=groups, class_balance=class_balance)
    clf.fit(features, labels, n_splits=n_splits, importance=importance)
    print(f'{len(clf.feature_names_)} features used')

    res = evaluate(clf.oof_)
    text = format_report(res, 'Out-of-fold, calibrated, manual labels')
    text += '\n' + format_report(evaluate(clf.oof_raw_), 'Out-of-fold, uncalibrated, manual labels')
    with open(f'{out_dir}/report.txt', 'w') as f:
        f.write(text)
    print('\n' + text)

    clf.oof_.to_csv(f'{out_dir}/oof_predictions.csv', index=False)
    plot_confusion(res['confusion'], f'{out_dir}/confusion.png')
    plot_reliability(clf.oof_raw_, clf.oof_, clf.classes_, f'{out_dir}/reliability.png')
    plot_p_astro(clf.oof_, clf.classes_, f'{out_dir}/p_astro_by_class.png')
    if importance and hasattr(clf, 'importance_'):
        clf.importance_.to_csv(f'{out_dir}/importance.csv', index=False)
        plot_importance(clf.importance_, f'{out_dir}/importance.png')

    if ablation:
        rows = []
        for subset in ABLATIONS:
            sub = EventClassifier(feature_groups=subset, class_balance=class_balance)
            sub.fit(features, labels, n_splits=n_splits, verbose=False)
            rows.append({'groups': '+'.join(subset), 'n_features': len(sub.feature_names_),
                         **summary_row(evaluate(sub.oof_))})
            print(f"  ablation {rows[-1]['groups']:28s} macro F1 {rows[-1]['macro_f1']:.3f}  AUC {rows[-1]['roc_auc']:.3f}")
        pd.DataFrame(rows).to_csv(f'{out_dir}/ablation.csv', index=False)
        print('\n' + pd.DataFrame(rows).round(3).to_string(index=False))

    if save_model:
        clf.save(save_model)

    if review or truth is not None:
        predictions = clf.predict(features)
        if review:
            rank_for_review(predictions, labels, n=review).to_csv(f'{out_dir}/review_queue.csv', index=False)
        if truth is not None:
            truth_check(predictions, labels, pd.read_csv(truth), f'{out_dir}/truth_check.txt')
    return clf


if __name__ == '__main__':
    run(FEATURES, SORT_DIR, OUT_DIR, pipeline_weight=PIPELINE_WEIGHT, crossbin_weight=CROSSBIN_WEIGHT, groups=GROUPS,
        class_balance=CLASS_BALANCE, n_splits=N_SPLITS, ablation=ABLATION, importance=IMPORTANCE,
        save_model=SAVE_MODEL, review=REVIEW, truth=TRUTH)

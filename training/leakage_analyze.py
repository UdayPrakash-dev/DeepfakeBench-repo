"""
Leakage / disentanglement analysis on features saved by leakage_extract.py.

  python training/leakage_analyze.py --feats ./leakage_feats/released.npz \
                                     --control ./leakage_feats/control.npz \
                                     --json ./leakage_feats/report.json

Part A  linear probes: how much real/fake (and manipulation-type) information can a LINEAR classifier
        read out of each branch?  Fit on the train split, scored on the test split.
Part B  dependence between branches (test split): linear CKA and distance correlation, each next to a
        permutation null (rows of one side shuffled), so the number is judged against "no relation".
Only needs numpy / scipy / scikit-learn (no GPU).
"""
import json
import argparse

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

BRANCHES = [('content', 'content  (encoder_c)'),
            ('forgery', 'forgery  (encoder_f)'),
            ('spe', 'specific f_spe'),
            ('sha', 'common   f_share')]
PAIRS = [('spe', 'sha'), ('content', 'sha'), ('content', 'spe'), ('content', 'forgery')]


# ---------------------------------------------------------------- Part A: probes
def probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    res = {}
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced')
    clf.fit(A, (ytr > 0).astype(int))
    res['bin_auc'] = float(roc_auc_score((yte > 0).astype(int), clf.predict_proba(B)[:, 1]))
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced')
    clf.fit(A, ytr)
    res['spe5_acc'] = float(accuracy_score(yte, clf.predict(B)))
    return res


# ------------------------------------------------------- Part B: dependence measures
def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, 'fro') ** 2
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return float(num / den) if den > 0 else 0.0


def _dcenter(D):
    return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()


def dcor(X, Y):
    A, B = _dcenter(cdist(X, X)), _dcenter(cdist(Y, Y))
    dxy, dxx, dyy = (A * B).mean(), (A * A).mean(), (B * B).mean()
    if dxx * dyy <= 0:
        return 0.0
    return float(np.sqrt(max(dxy, 0.0) / np.sqrt(dxx * dyy)))


def with_null(fn, X, Y, rng, n_perm):
    obs = fn(X, Y)
    null = np.array([fn(X, Y[rng.permutation(len(Y))]) for _ in range(n_perm)])
    return obs, float(null.mean()), float(null.std())


def zs(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--control', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--max_n', type=int, default=1500, help='subsample size for dependence measures')
    ap.add_argument('--perms', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    R = np.load(args.feats, allow_pickle=True)
    C = np.load(args.control, allow_pickle=True) if args.control else None
    report = {'probes': {}, 'dependence': {}}

    ytr, yte = R['train_y_spe'], R['test_y_spe']
    chance5 = float(np.bincount(yte).max() / len(yte))
    print(f"train n={len(ytr)}  test n={len(yte)}   test class counts (real,DF,F2F,FS,NT) = {np.bincount(yte).tolist()}")
    print(f"model's own common-head AUC on test subset: {roc_auc_score(yte > 0, R['test_prob']):.4f}\n")

    # ---- Part A
    print('=== A. Linear probes (fit on train split, scored on test split) ===')
    hdr = f"{'branch':22s} {'real/fake AUC':>14s} {'5-way acc':>10s}"
    if C is not None:
        hdr += f" | {'control AUC':>12s} {'control acc':>12s}"
    print(hdr)
    for key, name in BRANCHES:
        r = probe(R[f'train_{key}'], ytr, R[f'test_{key}'], yte)
        report['probes'][key] = {'released': r}
        line = f"{name:22s} {r['bin_auc']:14.3f} {r['spe5_acc']:10.3f}"
        if C is not None:
            c = probe(C[f'train_{key}'], C['train_y_spe'], C[f'test_{key}'], C['test_y_spe'])
            report['probes'][key]['control'] = c
            line += f" | {c['bin_auc']:12.3f} {c['spe5_acc']:12.3f}"
        print(line)
    print(f"(chance: real/fake AUC 0.500, 5-way accuracy {chance5:.3f};"
          f" control = same encoders at ImageNet init, never trained on deepfakes)\n")

    # ---- Part B
    print('=== B. Dependence between branches (test split) ===')
    n = len(yte)
    for subset_name, mask in [('all images', np.ones(n, bool)), ('fake images only', yte > 0)]:
        idx = np.where(mask)[0]
        if len(idx) > args.max_n:
            idx = rng.choice(idx, args.max_n, replace=False)
        print(f'-- {subset_name} (n={len(idx)})')
        print(f"{'pair':22s} {'measure':>10s} {'observed':>9s} {'null mean':>10s} {'null sd':>8s} {'excess':>8s}")
        for a, b in PAIRS:
            X, Y = zs(R[f'test_{a}'][idx]), zs(R[f'test_{b}'][idx])
            for mname, fn in [('lin. CKA', linear_cka), ('dist.corr', dcor)]:
                obs, nm, ns = with_null(fn, X, Y, rng, args.perms)
                print(f"{a + ' vs ' + b:22s} {mname:>10s} {obs:9.3f} {nm:10.3f} {ns:8.3f} {obs - nm:8.3f}")
                report['dependence'].setdefault(subset_name, {}).setdefault(f'{a}_vs_{b}', {})[mname] = {
                    'observed': obs, 'null_mean': nm, 'null_sd': ns}
        print()

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
        print('saved', args.json)


if __name__ == '__main__':
    main()

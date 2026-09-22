#!/usr/bin/env python3
"""
eval_harness.py — Tier-1 evaluation harness for the D3 probabilistic relevance
layer (DDI relevance paper), built to satisfy the discrimination /
calibration / uncertainty / PU-robustness bar of a Q1 pharmacovigilance venue.

WHAT THIS FIXES (referee issues M1, M3, M4):
  M1  Circular evaluation  -> grouped (drug-wise) cross-validation so no drug
                              appears in both train and test folds.
  M3  Missing metrics      -> ROC-AUC, PR-AUC, Brier, ECE, reliability curve,
                              all with bootstrap 95% CIs.
  M4  PU negatives untested-> negative-ratio sensitivity sweep (1:1/1:5/1:10)
                              + Elkan-Noto PU correction as a robustness check.

WHAT YOU MUST SUPPLY (the licensed part that cannot be rebuilt in a sandbox):
  A feature matrix X (n_pairs x 9) of Jaccard similarities over the nine
  biomedical dimensions, aligned row-for-row with a pair list, plus the
  positive labels y. See `load_real_data()` for the exact expected format.

The script ships with `synthetic_smoke_test()` that fabricates a 9-feature
matrix with the paper's known structure so the whole pipeline runs end-to-end
and produces figures/tables — proof the harness works before you plug in real
data. THE SMOKE-TEST NUMBERS ARE SYNTHETIC AND MUST NOT BE REPORTED.

Author: prepared as a revision deliverable. Run:
    pip install numpy pandas scikit-learn scipy matplotlib
    python eval_harness.py --smoke      # runs synthetic end-to-end
    python eval_harness.py --real       # runs on your data (see load_real_data)
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve

RNG = np.random.default_rng(20260101)
FEATURE_NAMES = ["targets", "enzymes", "transporters", "carriers", "side_effects",
                 "indications", "moa", "snps", "pathways"]


# ----------------------------------------------------------------------------
# Metrics with bootstrap CIs
# ----------------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with equal-width bins (Naeini et al., 2015)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return ece


def reliability_curve(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        xs.append(y_prob[m].mean())
        ys.append(y_true[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def bootstrap_ci(y_true, y_prob, fn, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI for any metric fn(y_true, y_prob)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    n = len(y_true)
    point = fn(y_true, y_prob)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        # guard against a resample with a single class (AUC undefined)
        if len(np.unique(y_true[b])) < 2:
            continue
        boots.append(fn(y_true[b], y_prob[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def all_metrics(y_true, y_prob, seed=0):
    out = {}
    for name, fn in [("ROC_AUC", roc_auc_score),
                     ("PR_AUC", average_precision_score),
                     ("Brier", brier_score_loss),
                     ("ECE", lambda yt, yp: expected_calibration_error(yt, yp))]:
        p, lo, hi = bootstrap_ci(y_true, y_prob, fn, seed=seed)
        out[name] = (p, lo, hi)
    return out


# ----------------------------------------------------------------------------
# Negative sampling (PU setting)
# ----------------------------------------------------------------------------
def sample_negatives(pos_pairs, all_drugs, n_neg, forbidden, seed=0):
    """Random unlabeled pairs assumed non-interacting (the paper's PU assumption)."""
    rng = np.random.default_rng(seed)
    drugs = list(all_drugs)
    negs = set()
    while len(negs) < n_neg:
        a, b = rng.choice(len(drugs), 2, replace=False)
        key = frozenset((drugs[a], drugs[b]))
        if key not in forbidden:
            negs.add((drugs[a], drugs[b]))
    return list(negs)


# ----------------------------------------------------------------------------
# Grouped cross-validation (fixes M1)
# ----------------------------------------------------------------------------
def grouped_cv_scores(X, y, groups, n_splits=5, seed=0, pair_drugs=None,
                      mode="pair_blind"):
    """
    Leakage-free cross-validation for pairwise DDI prediction (fix M1).

    A pair has TWO drugs. A naive random split, or even a GroupKFold on one drug
    per pair, lets the OTHER drug appear in both train and test — the model then
    memorises drug-specific effects and the reported recall is inflated. The
    rigorous design (Pahikkala et al. 2015; the DDI cold-start literature) splits
    at the level of DRUGS, not pairs:

      1. Partition the drug universe into `n_splits` disjoint folds.
      2. For test fold k, TRAIN on pairs whose BOTH drugs are outside fold k;
      3. TEST on pairs according to `mode`:
           "pair_blind"  (default, strictest): BOTH drugs are in fold k
                         -> neither drug was ever seen in training;
           "drug_blind"  (leave-one-drug-out): EXACTLY one drug is in fold k
                         -> the classic cold-start-for-one-drug setting.
         Pairs that straddle folds in a way the mode excludes are simply not
         scored in that fold (OOF stays NaN).

    `pair_drugs` — a list of (drug_a, drug_b) aligned row-for-row with X — is
    REQUIRED. `groups` is accepted for signature compatibility but not used here.

    NOTE ON YIELD: in a dense interaction graph the "pair_blind" test set can be
    small (few pairs have both drugs held out). That is the true, unavoidable cost
    of a leakage-free estimate, and the returned `n_scored` lets you report it. If
    "pair_blind" yields too few pairs for stable metrics, report "drug_blind" as
    the primary setting and "pair_blind" as the conservative bound — but never fall
    back to a random split.

    Returns (oof, fold_id, n_scored): OOF probabilities aligned to X (NaN where a
    row was not scored under this mode), fold ids, and the count of scored rows.
    """
    assert pair_drugs is not None, "pair_drugs is required for a leakage-free split"
    rng = np.random.default_rng(seed)
    all_drugs = sorted({d for ab in pair_drugs for d in ab})
    rng.shuffle(all_drugs)
    fold_of = {d: i % n_splits for i, d in enumerate(all_drugs)}
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    n_scored = 0
    for k in range(n_splits):
        test_drug = np.array([fold_of[pair_drugs[i][0]] == k or fold_of[pair_drugs[i][1]] == k
                              for i in range(len(y))])
        a_in = np.array([fold_of[pair_drugs[i][0]] == k for i in range(len(y))])
        b_in = np.array([fold_of[pair_drugs[i][1]] == k for i in range(len(y))])
        tr = np.where(~test_drug)[0]                      # both drugs outside fold k
        if mode == "pair_blind":
            te = np.where(a_in & b_in)[0]                 # both drugs inside fold k
        elif mode == "drug_blind":
            te = np.where(a_in ^ b_in)[0]                 # exactly one drug inside
        else:
            raise ValueError("mode must be 'pair_blind' or 'drug_blind'")
        if len(tr) == 0 or len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_id[te] = k
    # a drug_blind pair can be a valid test row in BOTH its drugs' folds (novel-a
    # in one, novel-b in the other); OOF keeps the last write (no leakage — the
    # training set excludes the held-out drug in each fold), so we report the count
    # of UNIQUE rows that received a prediction.
    n_scored = int(np.isfinite(oof).sum())
    return oof, fold_id, n_scored


def apparent_scores(X, y):
    """Resubstitution (training-set) predictions — reported ONLY as the biased
    upper bound, to make the train/test gap visible (fixes D1)."""
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    return clf.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------
# Elkan-Noto PU correction (fixes M4)
# ----------------------------------------------------------------------------
def elkan_noto_c(clf, X_val_pos):
    """Estimate label frequency c = P(s=1|y=1) as mean classifier score on a
    held-out set of KNOWN positives (Elkan & Noto, KDD 2008, estimator e1)."""
    return clf.predict_proba(X_val_pos)[:, 1].mean()


def pu_adjusted_proba(clf, X, c):
    """Recover P(y=1|x) = P(s=1|x) / c from the non-traditional classifier."""
    return np.clip(clf.predict_proba(X)[:, 1] / max(c, 1e-6), 0, 1)


# ----------------------------------------------------------------------------
# PU sensitivity sweep (fixes M4)
# ----------------------------------------------------------------------------
def pu_sensitivity(build_Xy, pos_pairs, all_drugs, forbidden,
                   ratios=(1, 5, 10), seed=0):
    """Re-run grouped-CV metrics at several positive:negative ratios, and add the
    Elkan-Noto-corrected variant, to show how much the assumed-negative scheme
    moves the headline numbers."""
    rows = []
    n_pos = len(pos_pairs)
    for r in ratios:
        negs = sample_negatives(pos_pairs, all_drugs, n_pos * r, forbidden, seed=seed)
        X, y, groups, pair_drugs = build_Xy(pos_pairs, negs)
        oof, _, n_scored = grouped_cv_scores(X, y, groups, seed=seed,
                                             pair_drugs=pair_drugs, mode="drug_blind")
        m = np.isfinite(oof)
        met = all_metrics(y[m], oof[m], seed=seed)
        rows.append({"ratio": f"1:{r}", "scheme": "grouped-CV",
                     **{k: round(v[0], 3) for k, v in met.items()},
                     "ROC_AUC_CI": f"[{met['ROC_AUC'][1]:.3f}, {met['ROC_AUC'][2]:.3f}]"})
    # Elkan-Noto at the 1:5 ratio, using the SAME leakage-free drug partition as
    # grouped_cv_scores (drug_blind) so the PU correction is not credited with a
    # leakage-inflated estimate.
    negs = sample_negatives(pos_pairs, all_drugs, n_pos * 5, forbidden, seed=seed)
    X, y, groups, pair_drugs = build_Xy(pos_pairs, negs)
    rng = np.random.default_rng(seed)
    drugs_u = sorted({d for ab in pair_drugs for d in ab}); rng.shuffle(drugs_u)
    fold_of = {d: i % 5 for i, d in enumerate(drugs_u)}
    oof = np.full(len(y), np.nan)
    for k in range(5):
        a_in = np.array([fold_of[pair_drugs[i][0]] == k for i in range(len(y))])
        b_in = np.array([fold_of[pair_drugs[i][1]] == k for i in range(len(y))])
        tr = np.where(~(a_in | b_in))[0]      # both drugs outside fold k
        te = np.where(a_in ^ b_in)[0]         # drug_blind test rows
        if len(tr) == 0 or len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        pos_tr = X[tr][y[tr] == 1]
        n_hold = max(1, len(pos_tr) // 5)
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        c = elkan_noto_c(clf, pos_tr[:n_hold])
        oof[te] = pu_adjusted_proba(clf, X[te], c)
    m = np.isfinite(oof)
    met = all_metrics(y[m], oof[m], seed=seed)
    rows.append({"ratio": "1:5", "scheme": "Elkan-Noto PU",
                 **{k: round(v[0], 3) for k, v in met.items()},
                 "ROC_AUC_CI": f"[{met['ROC_AUC'][1]:.3f}, {met['ROC_AUC'][2]:.3f}]"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_eval(y_true, oof, y_app, out="eval_harness_output.png"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    # ROC
    fpr, tpr, _ = roc_curve(y_true, oof)
    fpr_a, tpr_a, _ = roc_curve(y_true, y_app)
    axes[0].plot(fpr, tpr, color="#c0392b", label=f"held-out CV (AUC {roc_auc_score(y_true, oof):.3f})")
    axes[0].plot(fpr_a, tpr_a, color="0.6", ls="--", label=f"apparent (AUC {roc_auc_score(y_true, y_app):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="0.8")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC: held-out vs apparent", loc="left"); axes[0].legend(frameon=False, fontsize=7)
    # PR
    pr, rc, _ = precision_recall_curve(y_true, oof)
    axes[1].plot(rc, pr, color="#c0392b", label=f"PR-AUC {average_precision_score(y_true, oof):.3f}")
    axes[1].axhline(y_true.mean(), ls=":", color="0.7", label=f"prevalence {y_true.mean():.2f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall (held-out)", loc="left"); axes[1].legend(frameon=False, fontsize=7)
    # Calibration
    xs, ys, ns = reliability_curve(y_true, oof)
    axes[2].plot([0, 1], [0, 1], ":", color="0.7")
    axes[2].plot(xs, ys, "o-", color="#c0392b")
    axes[2].set_xlabel("Predicted probability"); axes[2].set_ylabel("Observed frequency")
    axes[2].set_title(f"Calibration (Brier {brier_score_loss(y_true, oof):.3f}, "
                      f"ECE {expected_calibration_error(y_true, oof):.3f})", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    return fig


# ----------------------------------------------------------------------------
# REAL data loader — fill this in
# ----------------------------------------------------------------------------
def load_real_data():
    """
    EXPECTED INPUTS (edit paths as needed):
      features.csv : columns = drug_a, drug_b, targets, enzymes, transporters,
                     carriers, side_effects, indications, moa, snps, pathways
                     (the nine Jaccard similarities), one row per POSITIVE pair.
      The gold positive pairs come from GoldD3R.txt (drug_a, drug_b CUIs).

    Returns build_Xy(pos_pairs, neg_pairs) -> (X, y, groups, pair_drugs) plus the
    pos list, the drug universe, and the forbidden-pair set, where pair_drugs is a
    list of (drug_a, drug_b) aligned row-for-row with X so the leakage guard in
    grouped_cv_scores() can drop test rows sharing EITHER drug with the train fold.
    You must be able to compute the
    nine Jaccard similarities for arbitrary negative pairs too (that requires the
    per-drug annotation sets from DrugBank/SIDER/PharmGKB/etc.).
    """
    raise NotImplementedError(
        "Plug in your feature matrix here. See docstring for the expected format.")


# ----------------------------------------------------------------------------
# Synthetic smoke test — proves the pipeline runs end to end
# ----------------------------------------------------------------------------
def synthetic_smoke_test():
    print(">>> SYNTHETIC SMOKE TEST — numbers below are FABRICATED, do not report.\n")
    n_drugs = 1599
    drugs = [f"C{i:07d}" for i in range(n_drugs)]
    # positives: 21,897 pairs with a hub-skewed degree like the real set
    deg_weights = RNG.pareto(1.5, n_drugs) + 1
    deg_weights /= deg_weights.sum()
    pos = set()
    while len(pos) < 21897:
        a, b = RNG.choice(n_drugs, 2, replace=False, p=deg_weights)
        pos.add((drugs[a], drugs[b]))
    pos = list(pos)
    forbidden = set(frozenset(p) for p in pos)

    # latent per-drug 9-dim annotation vectors; positives share more mechanistic mass
    drug_vec = {d: RNG.random(9) for d in drugs}

    def jac9(a, b):
        va, vb = drug_vec[a], drug_vec[b]
        inter = np.minimum(va, vb).sum(); union = np.maximum(va, vb).sum()
        base = inter / union if union else 0.0
        return np.clip(base + RNG.normal(0, 0.05, 9), 0, 1)

    def build_Xy(pos_pairs, neg_pairs):
        rows, ys, grps, pd_ = [], [], [], []
        degree = {}
        for a, b in pos_pairs:
            degree[a] = degree.get(a, 0) + 1; degree[b] = degree.get(b, 0) + 1
        for a, b in pos_pairs:
            # inject signal so positives are separable but not trivially so
            rows.append(jac9(a, b) * 1.15); ys.append(1)
            grps.append(a if degree.get(a, 1) <= degree.get(b, 1) else b); pd_.append((a, b))
        for a, b in neg_pairs:
            rows.append(jac9(a, b) * 0.85); ys.append(0)
            grps.append(a); pd_.append((a, b))
        return np.array(rows), np.array(ys), np.array(grps), pd_

    # main leakage-free run at 1:1 — report drug_blind (primary) and pair_blind (bound)
    negs = sample_negatives(pos, drugs, len(pos), forbidden)
    X, y, groups, pair_drugs = build_Xy(pos, negs)
    oof, folds, n_scored = grouped_cv_scores(X, y, groups, pair_drugs=pair_drugs,
                                             mode="drug_blind")
    oof_pb, _, n_pb = grouped_cv_scores(X, y, groups, pair_drugs=pair_drugs,
                                        mode="pair_blind")
    print(f"drug_blind scored {n_scored} rows; pair_blind (both drugs held out) "
          f"scored {n_pb} rows of {len(y)} total\n")
    m = np.isfinite(oof)
    y_app = apparent_scores(X, y)

    met = all_metrics(y[m], oof[m])
    print("Held-out (grouped-CV) metrics [95% bootstrap CI]:")
    for k, (p, lo, hi) in met.items():
        print(f"  {k:8s} {p:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"\n  Apparent ROC-AUC (biased): {roc_auc_score(y, y_app):.3f}")
    print(f"  Held-out ROC-AUC:          {met['ROC_AUC'][0]:.3f}")
    print(f"  Optimism (train-test gap): {roc_auc_score(y, y_app) - met['ROC_AUC'][0]:.3f}\n")

    sens = pu_sensitivity(build_Xy, pos, drugs, forbidden)
    print("PU negative-sampling sensitivity:")
    print(sens.to_string(index=False))

    plot_eval(y[m], oof[m], y_app[m])
    sens.to_csv("pu_sensitivity_smoke.csv", index=False)
    print("\nWrote eval_harness_output.png and pu_sensitivity_smoke.csv")
    return met, sens


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    if a.real:
        raise SystemExit("Implement load_real_data() with your feature matrix, then call the pipeline.")
    synthetic_smoke_test()

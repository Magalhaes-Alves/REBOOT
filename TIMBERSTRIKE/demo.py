"""End-to-end demonstration of TimberStrike against LightGBM.

The demo:
  1. trains a small LightGBM binary classifier on a synthetic dataset;
  2. runs TimberStrike to reconstruct the training set from the booster;
  3. reports the Reconstruction Accuracy (RA) at several tolerance levels.

Run it directly:

    python demo.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import lightgbm as lgb

from timberstrike_lgb import TimberStrikeLightGBM, reconstruction_accuracy


def make_dataset(n: int = 150, d: int = 6, seed: int = 7):
    rng = np.random.RandomState(seed)
    # Mix of normal and uniform features so we can study RA on both
    # heavy-tailed and bounded distributions.
    X = np.column_stack([
        rng.normal(0, 1, size=n),
        rng.normal(0, 1, size=n),
        rng.uniform(-2, 2, size=n),
        rng.uniform(0, 5, size=n),
        rng.normal(1, 0.5, size=n),
        rng.normal(-1, 1.5, size=n),
    ])[:, :d]
    # An interpretable target: linear combination + noise -> binary label.
    score = X[:, 0] + 0.7 * X[:, 1] - 0.3 * X[:, 2] + 0.2 * X[:, 3]
    y = (score + rng.normal(0, 0.3, size=n) > 0).astype(int)
    return X, y


def main() -> None:
    X, y = make_dataset()
    n, d = X.shape
    print(f"Training set: n={n} samples, d={d} features, "
          f"positive rate={y.mean():.2f}")

    lr = 0.3
    lam = 1.0
    booster = lgb.train(
        params=dict(
            objective="binary",
            learning_rate=lr,
            num_leaves=8,
            max_depth=4,
            min_data_in_leaf=1,
            lambda_l2=lam,
            feature_pre_filter=False,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        ),
        train_set=lgb.Dataset(X, label=y),
        num_boost_round=15,
    )

    # The attacker bounds every feature by its observed range across the
    # population.  In the federated setting this is public domain
    # knowledge (acceptable feature ranges are usually shared as the
    # schema of the FL protocol).
    feature_bounds = [
        (float(np.min(X[:, f]) - 0.5), float(np.max(X[:, f]) + 0.5))
        for f in range(d)
    ]

    attacker = TimberStrikeLightGBM(
        booster=booster,
        n_features=d,
        feature_bounds=feature_bounds,
        # base_score auto-recovered from the first tree
        learning_rate=lr,
        reg_lambda=lam,
        milp_time_limit=60,
        verbose=True,
    )
    X_rec, y_rec = attacker.attack()
    print(f"\nReconstructed n={len(X_rec)} samples (truth: {n}).")
    print(f"Recovered base_score = {attacker.base_score:.4f} "
          f"(true: {y.mean():.4f})")

    print("\nReconstruction Accuracy (overall and per feature):")
    for tol in (0.01, 0.05, 0.10):
        acc, per_feat = reconstruction_accuracy(
            X, X_rec, tol=tol, feature_ranges=feature_bounds
        )
        per_feat_str = " ".join(f"{a:.2f}" for a in per_feat)
        print(f"  tol={tol:.2f}  RA={acc*100:5.2f}%   "
              f"per-feature: [{per_feat_str}]")

    # Label-level reconstruction quality (matched via the same Hungarian
    # assignment as RA on the inputs).
    from scipy.optimize import linear_sum_assignment
    spans = np.array([hi - lo for lo, hi in feature_bounds])
    cost = np.zeros((n, len(X_rec)))
    for i in range(n):
        cost[i] = np.sum(np.abs(X_rec / spans - (X[i] / spans)), axis=1)
    row, col = linear_sum_assignment(cost)
    k = min(len(row), n, len(X_rec))
    label_acc = float(np.mean(y[row[:k]] == y_rec[col[:k]]))
    print(f"\nLabel reconstruction accuracy (after Hungarian match): "
          f"{label_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

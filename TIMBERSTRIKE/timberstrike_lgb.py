from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pulp


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #
@dataclass
class TreeNode:
    """In-memory representation of a LightGBM tree node."""
    is_leaf: bool
    # internal-node fields
    split_feature: Optional[int] = None
    threshold: Optional[float] = None
    decision_type: Optional[str] = None       
    default_left: bool = True
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    internal_count: Optional[int] = None
    internal_weight: Optional[float] = None   
    internal_value: Optional[float] = None

    leaf_index: Optional[int] = None
    leaf_value: Optional[float] = None        
    leaf_count: Optional[int] = None          
    leaf_weight: Optional[float] = None       


@dataclass
class LeafInfo:
    """Leaf together with the path constraints that lead to it."""
    leaf_index: int
    leaf_value: float
    N: int                                    # N_j (from leaf_count)
    H: float                                  # H_j (from leaf_weight)
    # path-implied feature bounds:
    #   lower[f] = strict lower bound (sample's feature value must be > lower[f])
    #   upper[f] = inclusive upper bound (sample's feature value must be <= upper[f])
    lower: Dict[int, float] = field(default_factory=dict)
    upper: Dict[int, float] = field(default_factory=dict)


@dataclass
class Sample:
    """A reconstructed sample's current state."""
    label: int                                # 0 or 1
    lo: np.ndarray                            # per-feature lower bound (inclusive)
    hi: np.ndarray                            # per-feature upper bound (inclusive)
    p: float                                  # current probability score
    g: float                                  # current gradient (p - y)
    h: float                                  # current Hessian   p*(1-p)
    leaf_per_tree: List[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  LightGBM dump parsing
# --------------------------------------------------------------------------- #
def _parse_tree(node: dict) -> TreeNode:
    """Recursively convert a LightGBM tree-structure dict into ``TreeNode``s."""
    if "leaf_index" in node:
        return TreeNode(
            is_leaf=True,
            leaf_index=int(node["leaf_index"]),
            leaf_value=float(node["leaf_value"]),
            leaf_count=int(node.get("leaf_count", 0)),
            leaf_weight=float(node.get("leaf_weight", 0.0)),
        )
    return TreeNode(
        is_leaf=False,
        split_feature=int(node["split_feature"]),
        threshold=float(node["threshold"]) if node["decision_type"] == "<="
                  else node["threshold"],   # categorical: string of bitset / values
        decision_type=node["decision_type"],
        default_left=bool(node.get("default_left", True)),
        left=_parse_tree(node["left_child"]),
        right=_parse_tree(node["right_child"]),
        internal_count=int(node.get("internal_count", 0)),
        internal_weight=float(node.get("internal_weight", 0.0)),
        internal_value=float(node.get("internal_value", 0.0)),
    )


def _collect_leaves(root: TreeNode) -> List[LeafInfo]:
    """Walk the tree depth-first, collecting one ``LeafInfo`` per leaf."""
    leaves: List[LeafInfo] = []

    def walk(node: TreeNode, lo: Dict[int, float], up: Dict[int, float]) -> None:
        if node.is_leaf:
            leaves.append(LeafInfo(
                leaf_index=node.leaf_index,
                leaf_value=node.leaf_value,
                N=node.leaf_count,
                H=node.leaf_weight,
                lower=dict(lo),
                upper=dict(up),
            ))
            return
        if node.decision_type != "<=":
            # categorical splits left for the extension note; for now treat
            # both children as path-unconstrained on the split feature
            walk(node.left, lo, up)
            walk(node.right, lo, up)
            return
        f, thr = node.split_feature, float(node.threshold)
        # left: feature <= thr
        new_up = dict(up)
        new_up[f] = min(new_up.get(f, math.inf), thr)
        walk(node.left, lo, new_up)
        # right: feature > thr (strict)
        new_lo = dict(lo)
        new_lo[f] = max(new_lo.get(f, -math.inf), thr)
        walk(node.right, new_lo, up)

    walk(root, {}, {})
    return leaves


def _sample_can_reach(sample: Sample, leaf: LeafInfo) -> bool:
    """True iff the sample's current box has non-empty intersection with the
    leaf's feature region."""
    for f, thr in leaf.upper.items():
        # leaf requires feature <= thr; sample must have some value <= thr
        # in its box, i.e. lo[f] <= thr
        if sample.lo[f] > thr:
            return False
    for f, thr in leaf.lower.items():
        # leaf requires feature > thr; sample must have some value > thr
        # in its box, i.e. hi[f] > thr
        if sample.hi[f] <= thr:
            return False
    return True


# --------------------------------------------------------------------------- #
#  The attack
# --------------------------------------------------------------------------- #
class TimberStrikeLightGBM:
    """TimberStrike dataset-reconstruction attack against a LightGBM
    binary classifier.

    Parameters
    ----------
    booster :
        A trained ``lightgbm.Booster`` (or any object exposing the same
        ``dump_model`` JSON layout).
    n_features :
        Number of input features.
    feature_bounds :
        Sequence of ``(low, high)`` pairs giving prior bounds on every
        feature.  Used wherever no split tightens the range.  If a feature
        is unbounded the attacker can pass a large finite interval (e.g.
        the observed min/max of any auxiliary distribution).
    base_score :
        Optional probability ``b`` such that the model's initial raw score
        equals ``logit(b)``.  When ``None`` the value is recovered
        automatically from the first tree's leaf-weight/leaf-count ratio.
    learning_rate, reg_lambda :
        The hyper-parameters used at training time.  In a federated
        setting the attacker knows them since they are part of the
        federation protocol.  ``reg_lambda`` corresponds to LightGBM's
        ``lambda_l2``.
    milp_time_limit :
        Per-tree time budget (seconds) given to the MILP solver -- the
        paper uses 600s (10 minutes).
    """

    def __init__(
        self,
        booster,
        n_features: int,
        feature_bounds: Sequence[Tuple[float, float]],
        base_score: Optional[float] = None,
        learning_rate: float = 0.1,
        reg_lambda: float = 0.0,
        milp_time_limit: int = 600,
        verbose: bool = False,
    ):
        self.booster = booster
        self.n_features = n_features
        if len(feature_bounds) != n_features:
            raise ValueError("feature_bounds must have length n_features")
        self.feature_bounds = [(float(lo), float(hi)) for k, [lo, hi] in feature_bounds.items()]
        self.learning_rate = float(learning_rate)
        self.reg_lambda = float(reg_lambda)
        self.milp_time_limit = int(milp_time_limit)
        self.verbose = verbose

        dump = booster.dump_model()
        if dump.get("objective", "").split()[0] != "binary":
            raise ValueError(
                "This implementation targets binary classification; got "
                f"objective={dump.get('objective')!r}"
            )
        self.dump = dump
        self.trees = [_parse_tree(t["tree_structure"]) for t in dump["tree_info"]]
        # cache the leaves of every tree so we can update statistics later
        self._leaves_per_tree: List[List[LeafInfo]] = [
            _collect_leaves(root) for root in self.trees
        ]
        # per-tree shrinkage from the dump (1.0 for tree 0 when
        # boost_from_average=True, learning_rate otherwise).
        self._shrinkage: List[float] = [
            float(t.get("shrinkage", self.learning_rate))
            for t in dump["tree_info"]
        ]
        # Detect whether LightGBM absorbed the initial logit into tree 0
        # (i.e. boost_from_average=True for binary objective).  This is
        # signalled by shrinkage == 1.0 on tree 0 while subsequent trees
        # use the actual learning rate.
        self.init_absorbed_in_tree0: bool = (
            len(self._shrinkage) > 0
            and abs(self._shrinkage[0] - 1.0) < 1e-9
            and (len(self._shrinkage) < 2
                 or abs(self._shrinkage[1] - self.learning_rate) < 1e-6)
        )

        # ------- recover / store base score ------- #
        if base_score is None:
            base_score = self._recover_base_score()
        if not (0.0 < base_score < 1.0):
            raise ValueError(f"base_score must lie in (0, 1); got {base_score}")
        self.base_score = float(base_score)
        self.init_score = math.log(self.base_score / (1.0 - self.base_score))

        self.samples: List[Sample] = []

    # ------------------------------------------------------------------ #
    #  Auxiliary helpers
    # ------------------------------------------------------------------ #
    def _recover_base_score(self) -> float:
        """Recover ``b`` from the first tree.

        On the very first tree every sample shares the same Hessian
        ``h_i = b*(1-b)``; the ratio ``H_j / N_j = b*(1-b)`` is therefore
        identical for every leaf of tree 0.  Solving the quadratic gives
        two candidate roots ``b = 0.5 * (1 ± sqrt(1 - 4*h))``; the
        ambiguity is resolved by exploiting Eq. (8)-(9):

            G_j = -leaf_value_fit * (H_j + lambda)        (tree-0 fit term)
            N^{(1)}_j = N_j * b - G_j                     (positives in leaf j)

        ``N^{(1)}_j`` must be (i) integer-valued and (ii) within
        ``[0, N_j]``.  Whichever root produces values most consistent
        with both constraints across all leaves of tree 0 is selected.

        The "fit term" depends on whether LightGBM absorbed the initial
        logit into tree 0:

            init_absorbed_in_tree0=True   (boost_from_average=True):
                leaf_value = logit(b) + lr * (-G_j / (H_j + lambda))
                => -G_j / (H_j + lambda) = (leaf_value - logit(b)) / lr

            init_absorbed_in_tree0=False  (boost_from_average=False):
                leaf_value = lr * (-G_j / (H_j + lambda))
                => -G_j / (H_j + lambda) =  leaf_value / lr
        """
        first = self._leaves_per_tree[0]
        ratios = [lf.H / lf.N for lf in first if lf.N > 0]
        h = float(np.median(ratios))
        disc = max(0.0, 1.0 - 4.0 * h)
        b_plus = 0.5 * (1.0 + math.sqrt(disc))
        b_minus = 0.5 * (1.0 - math.sqrt(disc))
        shr0 = self._shrinkage[0]
        # When the dump has shrinkage=1 on tree 0 we believe init is
        # absorbed; otherwise it isn't.
        init_absorbed = abs(shr0 - 1.0) < 1e-9 and (
            len(self._shrinkage) < 2
            or abs(self._shrinkage[1] - self.learning_rate) < 1e-6
        )

        def score(b: float) -> float:
            """Lower is better: sum of integer-rounding error of N_pos plus
            heavy penalty for leaves with N_pos outside [-0.5, N_j+0.5]."""
            if not (0.0 < b < 1.0):
                return math.inf
            logit_b = math.log(b / (1.0 - b))
            total = 0.0
            for lf in first:
                if lf.N <= 0:
                    continue
                # invert leaf_value -> G_j depending on init absorption
                if init_absorbed:
                    fit = (lf.leaf_value - logit_b) / self.learning_rate
                else:
                    fit = lf.leaf_value / shr0
                G = -fit * (lf.H + self.reg_lambda)
                n_pos = lf.N * b - G
                # range penalty
                if n_pos < -0.5 or n_pos > lf.N + 0.5:
                    total += 1e6 + abs(n_pos - max(0.0, min(lf.N, n_pos)))
                # integrality residual
                total += abs(n_pos - round(n_pos))
            return total

        scores = [(score(b_plus), b_plus), (score(b_minus), b_minus)]
        scores.sort(key=lambda t: t[0])
        return scores[0][1]

    def _initial_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([fb[0] for fb in self.feature_bounds], dtype=float)
        hi = np.array([fb[1] for fb in self.feature_bounds], dtype=float)
        return lo, hi

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    # ------------------------------------------------------------------ #
    #  PHASE 1 -- First-Tree Probing  (paper §5.1)
    # ------------------------------------------------------------------ #
    def first_tree_probing(self) -> None:
        b = self.base_score
        leaves = self._leaves_per_tree[0]
        shr0 = self._shrinkage[0]
        # The "fit value" of a tree-0 leaf is its post-shrinkage Newton
        # step  -G_j / (H_j + lambda).  Depending on whether LightGBM
        # absorbed the initial logit into tree 0, this is recovered as:
        #   * init absorbed (shr0=1):  fit = (leaf_value - logit(b)) / lr
        #   * init not absorbed     :  fit =  leaf_value / shr0
        def _fit_value(lf: LeafInfo) -> float:
            if self.init_absorbed_in_tree0:
                return (lf.leaf_value - self.init_score) / self.learning_rate
            return lf.leaf_value / shr0

        for leaf in leaves:
            if leaf.N <= 0:
                continue

            # Eq. (5)-(6): H_j is directly exposed by LightGBM, so N_j is
            # exact (we keep both checks for fidelity to the paper).
            N_j = leaf.N
            H_j = leaf.H

            # Eq. (7): G_j follows from inverting the leaf's update rule.
            G_j = -_fit_value(leaf) * (H_j + self.reg_lambda)

            # Eq. (8)-(9): label distribution
            #   G_j = N^{(0)}_j * b - N^{(1)}_j * (1 - b)  with N^{(0)} + N^{(1)} = N_j
            #   =>  N^{(1)} = N_j * b - G_j
            n_pos_real = N_j * b - G_j
            n_pos = int(round(n_pos_real))
            n_pos = max(0, min(N_j, n_pos))
            n_neg = N_j - n_pos

            # Build the path-implied feature box
            lo, hi = self._initial_bounds()
            for f, thr in leaf.lower.items():
                if thr > lo[f]:
                    lo[f] = thr           # strict ">" — handled as midpoint later
            for f, thr in leaf.upper.items():
                if thr < hi[f]:
                    hi[f] = thr

            # After first tree:
            #   raw_score = leaf_value      (no separate init term -- LightGBM
            #                                bakes init into tree 0 when
            #                                init_absorbed_in_tree0 is True,
            #                                otherwise the model starts at 0).
            raw0 = leaf.leaf_value if self.init_absorbed_in_tree0 \
                                   else (self.init_score + leaf.leaf_value)
            p = self._sigmoid(raw0)
            for y in [1] * n_pos + [0] * n_neg:
                self.samples.append(Sample(
                    label=y,
                    lo=lo.copy(),
                    hi=hi.copy(),
                    p=p,
                    g=p - y,
                    h=p * (1.0 - p),
                    leaf_per_tree=[leaf.leaf_index],
                ))

        if self.verbose:
            print(
                f"[Phase 1] First-Tree Probing: {len(self.samples)} samples "
                f"reconstructed from {sum(1 for lf in leaves if lf.N > 0)} leaves "
                f"(base_score={self.base_score:.4f}, "
                f"init_absorbed_in_tree0={self.init_absorbed_in_tree0})."
            )

    # ------------------------------------------------------------------ #
    #  PHASE 2 -- Feature-Range Inference  (paper §5.2)
    # ------------------------------------------------------------------ #
    def feature_range_inference(self) -> None:
        n_trees = len(self.trees)
        for t_idx in range(1, n_trees):
            t0 = time.time()
            self._refine_with_tree(t_idx)
            self._update_statistics_with_tree(t_idx)
            if self.verbose:
                print(
                    f"[Phase 2] tree {t_idx + 1}/{n_trees} processed in "
                    f"{time.time() - t0:.1f}s"
                )

    # ------------------------------------------------------------------ #
    def _refine_with_tree(self, t_idx: int) -> None:
        """Build and solve the per-tree MILP that assigns reconstructed
        samples to leaves, then tighten each sample's feature box.

        The paper's quadratic objective (Eq. 13)

            min  sum_j ( (sum_i x_ij g_i) - G_j )^2
                     + ( (sum_i x_ij h_i) - H_j )^2

        is linearised here as an L1 objective by introducing non-negative
        deviations ``dG+_j, dG-_j, dH+_j, dH-_j``.  This is the formulation
        actually compatible with the off-the-shelf CBC MILP solver
        bundled with PuLP and matches the "MILP" terminology used
        throughout the paper.
        """
        leaves = self._leaves_per_tree[t_idx]
        # Drop empty leaves (LightGBM occasionally produces leaves with 0 count
        # under aggressive regularisation; the attack ignores them).
        leaves = [lf for lf in leaves if lf.N > 0]
        if not leaves:
            return
        n = len(self.samples)
        m = len(leaves)

        # ------- pre-compute targets G_j, H_j ------- #
        # H_j is exposed directly.  G_j follows from Eq. (7), inverted with
        # the *per-tree* shrinkage from the dump (so this works correctly
        # even on tree 0 if we ever call this on it).
        shr = self._shrinkage[t_idx]
        G_target = np.array([
            -lf.leaf_value * (lf.H + self.reg_lambda) / shr
            for lf in leaves
        ])
        H_target = np.array([lf.H for lf in leaves])

        # ------- reachability sets L_i ------- #
        reachable: List[List[int]] = []
        for s in self.samples:
            li = [k for k, lf in enumerate(leaves) if _sample_can_reach(s, lf)]
            if not li:
                # The sample's box has been tightened past every leaf of
                # this tree (can happen with shallow trees + boundary
                # mid-points).  Fall back to "all leaves" so the MILP
                # remains feasible.
                li = list(range(m))
            reachable.append(li)

        # ------- build MILP ------- #
        prob = pulp.LpProblem(f"TimberStrike_t{t_idx}", pulp.LpMinimize)

        x: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for i in range(n):
            for k in reachable[i]:
                x[(i, k)] = pulp.LpVariable(f"x_{i}_{k}", cat="Binary")

        # one leaf per sample
        for i in range(n):
            prob += pulp.lpSum(x[(i, k)] for k in reachable[i]) == 1

        # L1 deviation variables for each leaf
        dGp = [pulp.LpVariable(f"dGp_{k}", lowBound=0) for k in range(m)]
        dGn = [pulp.LpVariable(f"dGn_{k}", lowBound=0) for k in range(m)]
        dHp = [pulp.LpVariable(f"dHp_{k}", lowBound=0) for k in range(m)]
        dHn = [pulp.LpVariable(f"dHn_{k}", lowBound=0) for k in range(m)]

        for k in range(m):
            samples_in_k = [(i, x[(i, k)]) for i in range(n) if k in reachable[i]]
            G_expr = pulp.lpSum(self.samples[i].g * v for i, v in samples_in_k)
            H_expr = pulp.lpSum(self.samples[i].h * v for i, v in samples_in_k)
            prob += G_expr - G_target[k] == dGp[k] - dGn[k]
            prob += H_expr - H_target[k] == dHp[k] - dHn[k]

        # objective: minimise the sum of absolute deviations (G and H are
        # on comparable scales for binary log-loss so equal weights are
        # appropriate; one can tune a weight if desired).
        prob += (pulp.lpSum(dGp) + pulp.lpSum(dGn)
                 + pulp.lpSum(dHp) + pulp.lpSum(dHn))

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.milp_time_limit)
        prob.solve(solver)

        # ------- harvest assignments, tighten boxes ------- #
        for i, s in enumerate(self.samples):
            chosen = None
            for k in reachable[i]:
                v = x[(i, k)].value()
                if v is not None and v > 0.5:
                    chosen = k
                    break
            if chosen is None:
                # Solver gave nothing for this sample; keep box unchanged.
                s.leaf_per_tree.append(-1)
                continue
            lf = leaves[chosen]
            for f, thr in lf.lower.items():
                if thr > s.lo[f]:
                    s.lo[f] = thr
            for f, thr in lf.upper.items():
                if thr < s.hi[f]:
                    s.hi[f] = thr
            s.leaf_per_tree.append(lf.leaf_index)

    # ------------------------------------------------------------------ #
    def _update_statistics_with_tree(self, t_idx: int) -> None:
        """After tree ``t_idx`` has been processed, recompute every sample's
        probability score, gradient and Hessian using the leaf values of all
        trees considered so far.  Implements Eq. (10).

        In LightGBM the raw prediction is simply ``sum_t leaf_value^(t)``
        when ``boost_from_average=True`` (the initial logit is baked into
        tree 0).  Otherwise we add ``init_score`` explicitly.
        """
        leaves = {lf.leaf_index: lf for lf in self._leaves_per_tree[t_idx]}
        for s in self.samples:
            leaf_idx = s.leaf_per_tree[-1]
            if leaf_idx < 0 or leaf_idx not in leaves:
                continue
            tree_sum = sum(
                self._leaves_per_tree[t][s.leaf_per_tree[t]].leaf_value
                for t in range(t_idx + 1)
                if s.leaf_per_tree[t] >= 0
            )
            raw = tree_sum if self.init_absorbed_in_tree0 \
                            else (self.init_score + tree_sum)
            s.p = self._sigmoid(raw)
            s.g = s.p - s.label
            s.h = s.p * (1.0 - s.p)

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #
    def reconstructed_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X_rec, y_rec)`` where every feature is the mid-point of
        its final inferred interval.  This is the standard read-out used by
        the paper's evaluation.
        """
        if not self.samples:
            return np.empty((0, self.n_features)), np.empty((0,), dtype=int)
        X = np.zeros((len(self.samples), self.n_features))
        y = np.zeros(len(self.samples), dtype=int)
        for i, s in enumerate(self.samples):
            X[i] = 0.5 * (s.lo + s.hi)
            y[i] = s.label
        return X, y

    def reconstructed_intervals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays ``(lo, hi)`` of shape ``(n_samples, n_features)``
        with the final feature intervals -- useful for analysis and for
        tolerance-based reconstruction-accuracy metrics."""
        lo = np.stack([s.lo for s in self.samples]) if self.samples else \
             np.empty((0, self.n_features))
        hi = np.stack([s.hi for s in self.samples]) if self.samples else \
             np.empty((0, self.n_features))
        return lo, hi

    # ------------------------------------------------------------------ #
    #  Convenience runner
    # ------------------------------------------------------------------ #
    def attack(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run both phases and return ``(X_rec, y_rec)``."""
        self.first_tree_probing()
        self.feature_range_inference()
        return self.reconstructed_dataset()


# --------------------------------------------------------------------------- #
#  Evaluation utilities (matching paper §6.1 "Reconstruction Accuracy")
# --------------------------------------------------------------------------- #
def reconstruction_accuracy(
    X_true: np.ndarray,
    X_rec: np.ndarray,
    tol: float = 0.05,
    feature_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[float, np.ndarray]:
    """Compute the Reconstruction Accuracy used in the TimberStrike paper.

    The accuracy is computed *after* solving the bipartite matching between
    true and reconstructed samples (Hungarian method, as in TabLeak).  A
    feature is considered correctly recovered when the absolute relative
    error is below ``tol`` (numerical features are normalised to
    ``[0, 1]`` using ``feature_ranges`` when supplied).

    Returns
    -------
    accuracy : float
        Overall percentage of correctly recovered (sample, feature) cells.
    per_feature : np.ndarray
        Per-feature accuracies (same metric, averaged across matched rows).
    """
    from scipy.optimize import linear_sum_assignment

    if X_true.shape[1] != X_rec.shape[1]:
        raise ValueError("X_true and X_rec must share the feature dimension")
    n_t, d = X_true.shape
    n_r = X_rec.shape[0]
    n = min(n_t, n_r)

    if feature_ranges is None:
        feature_ranges = [
            (float(np.min(X_true[:, f])), float(np.max(X_true[:, f])))
            for f in range(d)
        ]
    spans = np.array([max(hi - lo, 1e-12) for lo, hi in feature_ranges])

    Xt = X_true / spans
    Xr = X_rec / spans
    # cost matrix: pairwise L1 distances (rows: true, cols: reconstructed)
    cost = np.zeros((n_t, n_r))
    for i in range(n_t):
        cost[i] = np.sum(np.abs(Xr - Xt[i]), axis=1)
    row, col = linear_sum_assignment(cost)
    # only the first n pairs are valid (one-to-one match on the smaller side)
    row, col = row[:n], col[:n]

    matched_true = X_true[row] / spans
    matched_rec = X_rec[col] / spans
    correct = np.abs(matched_true - matched_rec) <= tol
    per_feature = correct.mean(axis=0)
    return float(correct.mean()), per_feature


# --------------------------------------------------------------------------- #
#  Notes on extensions
# --------------------------------------------------------------------------- #
# CATEGORICAL FEATURES.
#   LightGBM serialises categorical splits with ``decision_type == "=="`` and
#   ``threshold`` set to a ``"v1||v2||..."`` string of allowed categories.
#   To support them, replace the per-feature ``(lo, hi)`` box with a
#   per-feature set of allowed categorical values, update ``_collect_leaves``
#   to intersect those sets along the path, and replace the mid-point
#   read-out with the most-likely value of the final set.  The MILP layer is
#   unchanged.
#
# MULTI-CLASS (One-vs-Rest).
#   LightGBM produces ``num_class`` trees per boosting iteration.  Following
#   Appendix E of the paper, run First-Tree Probing on the first K trees,
#   using ``p_i^{(c)} = 1/K`` and ``h_i^{(c)} = 2 p (1-p)``; afterwards
#   process each subsequent boosting iteration as K independent MILPs, then
#   update per-sample softmax statistics jointly.

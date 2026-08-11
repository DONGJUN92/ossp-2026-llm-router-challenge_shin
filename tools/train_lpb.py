# SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
# SPDX-License-Identifier: Apache-2.0

"""Fit the router artifact from public Train, then size the tier margins.

    PYTHONPATH=src python3 tools/train_lpb.py \
        --train-input data/materialized/train/inputs.json \
        --train-outcomes data/train/outcomes.json \
        --artifact src/ossp_router/resources/lpb-artifact.v1.json

NumPy is used here and only here; the runtime in ``ossp_router.lpb`` is
standard library only, so no third-party package enters the submission image.

Three things are fitted:

*score* — a 2PL item-response model, ``p_m = sigmoid(a_m * (theta(x) - b_m))``
with a shared prompt ability ``theta(x) = w . phi(x)`` and two parameters per
model. Three independent regression heads were measured against this on grouped
5-fold CV over three fold seeds: the low-rank form won every seed and its score
varied by 0.0011 across seeds against 0.0230 for independent heads. With 1,760
training items and only three models, the shared-ability constraint is what
keeps the predictor stable, and stability is what a hidden evaluation set with
an undisclosed mixture rewards.

*tokens* — input and output counts per model, regressed in log space and
corrected by Duan's smearing factor. Cost is then assembled from the published
rates rather than regressed directly, because the rates are known exactly and
only the token counts are uncertain.

*tier margins* — the fraction of each tier's excess budget the router may plan
to spend. Chosen by routing held-out folds the cost model never saw and reading
the realised ratio, then backing off until the out-of-fold tail stays clear of
the limit. It is deliberately not chosen by maximising the public score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.lpb import (  # noqa: E402
    N_DENSE,
    _enforce_monotone_cost as enforce_monotone_cost,
    _ladder as lpb_ladder,
    content_key,
    featurize,
)
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

MODELS = ("ax31-light", "ax31", "axk1-think")


def load_pairs(input_path: Path, outcomes_path: Path):
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    by_key = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    rows = []
    for ep in inputs.episodes:
        got = [by_key.get((ep.episode_id, m)) for m in MODELS]
        if any(g is None for g in got):
            continue
        rows.append((episode_text(ep), got))
    return rows


def design(texts: Sequence[str], dim: int) -> np.ndarray:
    return np.asarray([featurize(t, dim) for t in texts], dtype=np.float64)


def ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def fit_2pl(X, Y, l2=3.0, iters=900, lr=0.5):
    """Shared ability theta(x) with per-model discrimination and difficulty."""
    n, p = X.shape
    m = Y.shape[1]
    w = np.zeros(p)
    la = np.zeros(m)
    b = np.zeros(m)
    for _ in range(iters):
        theta = X @ w
        a = np.exp(la)
        P = 1.0 / (1.0 + np.exp(-np.clip(a[None, :] * (theta[:, None] - b[None, :]), -30, 30)))
        D = (P - Y) / n
        w -= lr * (X.T @ (D * a[None, :]).sum(axis=1) + l2 / n * w)
        la -= lr * (D * (theta[:, None] - b[None, :]) * a[None, :]).sum(axis=0)
        b -= lr * (D * (-a[None, :])).sum(axis=0)
    return w, np.exp(la), b


def token_ranges(T: np.ndarray) -> List[Tuple[float, float]]:
    """Observed token box per model, used to clamp the log-linear extrapolation."""
    return [(float(T[:, j].min()), float(T[:, j].max())) for j in range(T.shape[1])]


C_PROBIT = math.pi / 8.0


def _softplus(x):
    return np.logaddexp(0.0, x)


class _Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= self.lr * (self.m[i] / (1 - 0.9 ** self.t)) / (
                np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8
            )


def fit_grm_items(Q, values, K=31, steps=400, lr=0.05):
    """Samejima graded response model, ability integrated out.

    MEASURED AND NOT ADOPTED. Reproduce with ``--predictor grm``.

    The ordinal model is the better description of the data -- ``score`` really
    does take five ordered values -- and it wins on mean absolute error
    (0.3042 against 0.3384) and on grouped-CV routing score (+0.069, 2.9x
    paired SE). On the held-out public Dev it nevertheless *loses*: 0.6715
    against 0.6822, and it never selects ``axk1-think`` at any tier, leaving
    Premium at ratio 1.69 of a 4.0 budget.

    The reason is a flaw in the CV metric, not in the model. A tier scores zero
    when it breaches its budget, so a predictor that rarely upgrades cannot
    breach and looks both better and far more stable. MML calibrates the items
    against a population ability; ``axk1-think`` comes out with discrimination
    1.63 against 3.83 and 4.73 for the others, so its expected score rarely
    exceeds ``ax31`` and the allocator is offered almost no upgrades to buy.
    Fitting ability and items jointly, as ``fit_2pl`` does, lets the items adapt
    to the variation the features can actually explain.

    ``score`` takes five ordered values (0, .25, .5, .75, 1) and 8% of the
    public grid sits strictly between 0 and 1. A binary item model has to round
    those away; the graded model puts a threshold between each pair of adjacent
    levels instead. Thresholds are kept ordered by construction (first threshold
    plus softplus increments), and ability is integrated over N(0,1) by
    Gauss-Hermite quadrature so the item parameters are not fitted jointly with
    a per-episode ability that would absorb them.
    """
    N, M = Q.shape
    L = len(values)
    g = np.abs(Q[..., None] - values[None, None, :]).argmin(-1)
    t, w = np.polynomial.hermite_e.hermegauss(K)
    logw = np.log(w / np.sqrt(2 * np.pi))
    raw_a = np.full(M, 0.55)
    b1 = np.zeros(M)
    raw_d = np.full((M, L - 2), -0.5) if L > 2 else np.zeros((M, 0))
    onehot = np.eye(L, dtype=bool)[g]
    opt = _Adam([raw_a, b1, raw_d], lr)

    def thresholds(b1_, raw_d_):
        if L == 2:
            return b1_[:, None]
        return np.concatenate(
            [b1_[:, None], b1_[:, None] + np.cumsum(_softplus(raw_d_), axis=1)], axis=1
        )

    for _ in range(steps):
        a = _softplus(raw_a)
        thr = thresholds(b1, raw_d)
        P = 1.0 / (1.0 + np.exp(-np.clip(a[None, :, None] * (t[:, None, None] - thr[None, :, :]), -30, 30)))
        cum = np.concatenate([np.ones((K, M, 1)), P, np.zeros((K, M, 1))], axis=2)
        pr = np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0)
        obs = np.transpose(np.stack([pr[:, m, g[:, m]] for m in range(M)], axis=2), (1, 0, 2))
        s = np.log(obs).sum(axis=2) + logw[None, :]
        mx = s.max(axis=1, keepdims=True)
        Ls = np.exp(s - mx)
        r = Ls / Ls.sum(axis=1, keepdims=True)
        inv = np.zeros((N, K, M, L))
        inv[onehot[:, None, :, :].repeat(K, axis=1)] = (1.0 / obs).ravel()
        dC = np.zeros((N, K, M, L + 1))
        dC[..., :-1] += inv
        dC[..., 1:] -= inv
        dC = dC[..., 1:-1]
        dP = P * (1.0 - P)
        wgt = np.einsum("nk,nkml->kml", r, dC)
        ga = (wgt * dP * (t[:, None, None] - thr[None, :, :])).sum(axis=(0, 2)) * (
            1.0 / (1.0 + np.exp(-raw_a))
        )
        gthr = -(wgt * dP).sum(axis=0) * a[:, None]
        gb1 = gthr.sum(axis=1)
        gd = (
            np.cumsum(gthr[:, ::-1], axis=1)[:, ::-1][:, 1:] * (1.0 / (1.0 + np.exp(-raw_d)))
            if L > 2
            else np.zeros((M, 0))
        )
        opt.step([-ga / N, -gb1 / N, -gd / N])
    return _softplus(raw_a), thresholds(b1, raw_d)


def grm_expected(X, Wmu, Ws, a, thr, values):
    """``X`` is the raw design matrix: featurize() already carries a constant
    column at index dim+15, so no extra intercept is appended here. Adding one
    would make the head vectors one longer than the runtime feature vector."""
    mu = X @ Wmu
    s = _softplus(X @ Ws)
    d = np.sqrt(1.0 + C_PROBIT * (a[None, :, None] ** 2) * (s[:, None, None] ** 2))
    z = a[None, :, None] * (mu[:, None, None] - thr[None, :, :]) / d
    P = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    n, M, _ = P.shape
    cum = np.concatenate([np.ones((n, M, 1)), P, np.zeros((n, M, 1))], axis=2)
    pr = np.clip(cum[..., :-1] - cum[..., 1:], 1e-12, 1.0)
    return (pr * values[None, None, :]).sum(axis=2), (mu, s, d, z, P)


def fit_grm_heads(X, Q, a, thr, values, steps=600, lr=0.05, l2=1e-4):
    """Amortise the ability with items frozen, against the predictive objective.

    Regressing features onto the posterior ability summaries instead was
    measured and lost badly (-0.085): it improves mean absolute error while
    compressing the per-model gaps the allocator actually ranks on. Fitting the
    heads to predict the observed scores keeps those gaps.
    """
    Wmu = np.zeros(X.shape[1])
    Ws = np.zeros(X.shape[1])
    opt = _Adam([Wmu, Ws], lr)
    dv = (values[:-1] - values[1:])[None, None, :]
    for _ in range(steps):
        E, (mu, s, d, z, P) = grm_expected(X, Wmu, Ws, a, thr, values)
        r = (E - Q) / Q.size
        dP = P * (1.0 - P)
        dz_dmu = a[None, :, None] / d
        dz_ds = -z * C_PROBIT * (a[None, :, None] ** 2) * s[:, None, None] / (d ** 2)
        gm = (r[:, :, None] * dv * dP * dz_dmu).sum(axis=(1, 2))
        gs = (r[:, :, None] * dv * dP * dz_ds).sum(axis=(1, 2)) / (1.0 + np.exp(-(X @ Ws)))
        opt.step([X.T @ gm + l2 * Wmu, X.T @ gs + l2 * Ws])
    return Wmu, Ws


def fit_tokens(X, T, alpha=3.0):
    """Log-space regression plus Duan's smearing factor, per model."""
    W, smear = [], []
    for j in range(T.shape[1]):
        y = np.log(np.maximum(T[:, j], 1.0))
        w = ridge(X, y, alpha)
        W.append(w)
        smear.append(float(np.mean(np.exp(y - X @ w))))
    return np.stack(W), smear


def cost_of(policy, model_id: str, tin: float, tout: float) -> float:
    r = policy.models[model_id]
    unit = float(policy.token_unit)
    return (
        float(r.fixed_cost)
        + tin * float(r.input_token_rate) / unit
        + tout * float(r.output_token_rate) / unit
    )


def envelope(costs, scores, names):
    opts = sorted(zip(costs, scores, names), key=lambda t: (t[0], -t[1]))
    hull = [opts[0]]
    for c, s, m in opts[1:]:
        if s <= hull[-1][1]:
            continue
        while len(hull) >= 2:
            c0, s0, _ = hull[-2]
            c1, s1, _ = hull[-1]
            if (s1 - s0) * (c - c1) <= (s - s1) * (c1 - c0):
                hull.pop()
            else:
                break
        hull.append((c, s, m))
    return hull


def route(texts, P, C, tier, policy, excess) -> List[str]:
    """Same allocator as the runtime, driven by supplied predictions."""
    n = len(texts)
    light = policy.light_model_id
    li = MODELS.index(light)
    ladder = lpb_ladder(policy, MODELS)
    plans = []
    for i in range(n):
        costs = list(C[i])
        enforce_monotone_cost(costs, ladder)
        plans.append(envelope(costs, list(P[i]), MODELS))
    base = float(np.sum(C[:, li]))
    limit = base * (1.0 + (float(policy.tiers[tier].budget_multiplier) - 1.0) * excess)
    ups = []
    for i, hull in enumerate(plans):
        key = content_key(texts[i])
        for k in range(1, len(hull)):
            dc = hull[k][0] - hull[k - 1][0]
            ds = hull[k][1] - hull[k - 1][1]
            if dc > 0 and ds > 0:
                ups.append((-ds / dc, key, k, i, dc))
    ups.sort()
    spent = sum(h[0][0] for h in plans)
    level = [0] * n
    for _, _, k, i, dc in ups:
        if level[i] != k - 1:
            continue
        if spent + dc <= limit:
            spent += dc
            level[i] = k
    return [plans[i][level[i]][2] for i in range(n)]


def grouped_folds(texts: Sequence[str], k: int, seed: int) -> List[List[int]]:
    """Fold on a digit-stripped prefix signature.

    DeepMind Mathematics, RuleTaker and GSM8K items are template-generated with
    substituted numbers. A random split puts near-duplicates on both sides and
    reports a held-out number that is not held out.
    """
    import re as _re

    groups: Dict[int, List[int]] = {}
    for i, t in enumerate(texts):
        s = _re.sub(r"\d+", "#", t.lower())
        s = _re.sub(r"[^a-z가-힣#\s]", " ", s)
        key = content_key(" ".join(s.split()[:40]))
        groups.setdefault(key, []).append(i)
    order = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    # Break ties between equal-sized groups differently per seed, so repeated
    # runs give genuinely different partitions. Without this every "seed"
    # returns the same folds and any variance estimate across seeds is zero by
    # construction rather than by agreement.
    rnd = np.random.default_rng(seed)
    rnd.shuffle(order)
    order.sort(key=len, reverse=True)
    folds: List[List[int]] = [[] for _ in range(k)]
    for grp in order:
        folds.sort(key=len)
        folds[0].extend(grp)
    return [sorted(f) for f in folds]


_HANGUL = re.compile(r"[가-힣]")
_MCQ_OPT = re.compile(r"\n\s*[A-D][.)]\s")
_RULE = re.compile(r"\b(visits|chases|eats|sees|likes)\b", re.I)
_CRUX = re.compile(r"def f\(|assert f\(")
_DMM = re.compile(
    r"(Let [a-z]\w*\(?[a-z]?\)? = |Solve |Differentiate |Calculate |Simplify |"
    r"Factor |Suppose |What is the [a-z]+ derivative|Round |Sort |Divide |Work out )"
)


def coarse_family(text: str) -> str:
    """Approximate source family, for offline resampling only.

    `DATA_LICENSES.md` names eleven public source families and the private
    mixture over them is deliberately undisclosed. Composition risk lives at
    this granularity, not at the template level: a Dirichlet over the ~1,500
    template signatures is numerically indistinguishable from a plain bootstrap
    and reports a 0% breach rate for margins that in fact breach one time in ten.

    This label never reaches the router. It is computed here, in the trainer,
    purely to redraw the family mixture when sizing a margin.
    """
    if _HANGUL.search(text):
        return "ko-mcq" if ("Question:" in text or _MCQ_OPT.search(text)) else "ko-reasoning"
    if _CRUX.search(text):
        return "code-exec"
    if len(text) > 4000:
        return "long-context"
    if _RULE.search(text[:400]) and "If someone" in text:
        return "rule-logic"
    if text.lstrip().startswith("Question:") and _MCQ_OPT.search(text):
        return "en-mcq"
    if _DMM.search(text[:400]) and len(text) < 800:
        return "symbolic-math"
    if len(text) < 1200 and re.search(r"how (many|much)|total|\$|calculate", text, re.I):
        return "word-math"
    return "unclassified"


def _template_groups(texts: Sequence[str]) -> List[List[int]]:
    """Index lists per coarse source family."""
    groups: Dict[str, List[int]] = {}
    for i, t in enumerate(texts):
        groups.setdefault(coarse_family(t), []).append(i)
    return [groups[k] for k in sorted(groups)]


def breach_rate_under_shift(
    texts, P, C, TRUE_C, policy, tier, excess, groups, *,
    worlds: int = 40, concentration: float = 4.0, seed: int = 0,
) -> float:
    """Fraction of resampled compositions where this margin exceeds the limit.

    The out-of-fold ratio answers "what would this margin have spent on public
    Train". It cannot answer "what will it spend on a set whose mixture nobody
    has disclosed", and `docs/DATA_CARD.md` says that mixture is not disclosed.
    Each world here redraws the weight of every template group from a Dirichlet
    and resamples episodes under it, so a margin that only survives the public
    proportions is visibly unsafe rather than silently so.
    """
    rng = np.random.default_rng(seed)
    limit = float(policy.tiers[tier].budget_multiplier)
    li = MODELS.index(policy.light_model_id)
    n = len(texts)
    # alpha proportional to the observed share, scaled by concentration: small
    # concentration allows a private set dominated by one family, large keeps it
    # near the public proportions.
    base = np.asarray([len(g) for g in groups], dtype=float)
    alpha = concentration * len(groups) * base / base.sum()
    breaches = 0
    for _ in range(worlds):
        w = rng.dirichlet(alpha)
        counts = rng.multinomial(n, w)
        idx = np.concatenate(
            [rng.choice(groups[g], size=c, replace=True) for g, c in enumerate(counts) if c]
        )
        sub_texts = [texts[i] for i in idx]
        picks = route(sub_texts, P[idx], C[idx], tier, policy, excess)
        chosen = [MODELS.index(m) for m in picks]
        spend = sum(TRUE_C[idx[t], chosen[t]] for t in range(len(idx)))
        base = float(TRUE_C[idx, li].sum())
        if base > 0 and spend / base > limit:
            breaches += 1
    return breaches / worlds


def _oof_predictions(X, Y, TIN, TOUT, texts, policy, args, k, seed, use_grm):
    """Held-out score and cost predictions over one grouped K-fold partition."""
    n = len(texts)
    P = np.zeros_like(Y)
    C = np.zeros_like(Y)
    for f in grouped_folds(texts, k, seed):
        hold = np.asarray(f)
        rest = np.asarray([i for i in range(n) if i not in set(f)])
        Win, sin_ = fit_tokens(X[rest], TIN[rest], args.alpha)
        Wout, sout = fit_tokens(X[rest], TOUT[rest], args.alpha)
        if use_grm:
            values = np.unique(Y[rest])
            a_i, thr_i = fit_grm_items(Y[rest], values)
            Wmu_i, Ws_i = fit_grm_heads(X[rest], Y[rest], a_i, thr_i, values)
            P[hold] = grm_expected(X[hold], Wmu_i, Ws_i, a_i, thr_i, values)[0]
        else:
            w_i, a_i, b_i = fit_2pl(X[rest], Y[rest], l2=args.alpha)
            th = X[hold] @ w_i
            P[hold] = 1.0 / (1.0 + np.exp(
                -np.clip(a_i[None, :] * (th[:, None] - b_i[None, :]), -30, 30)))
        rin, rout = token_ranges(TIN[rest]), token_ranges(TOUT[rest])
        for j in range(len(MODELS)):
            tin = np.clip(np.exp(np.minimum(X[hold] @ Win[j], 20)) * sin_[j], *rin[j])
            tout = np.clip(np.exp(np.minimum(X[hold] @ Wout[j], 20)) * sout[j], *rout[j])
            C[hold, j] = [cost_of(policy, MODELS[j], tin[t], tout[t])
                          for t in range(len(hold))]
    return P, C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-input", type=Path, required=True)
    ap.add_argument("--train-outcomes", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--margin-folds", type=int, default=12,
                    help="마진 산정용 fold 수. 배포 모델은 Train 전체로 적합되므로 "
                         "fold 수가 적으면 배포보다 나쁜 예측기로 마진을 재게 된다")
    ap.add_argument("--margin-repeats", type=int, default=2,
                    help="마진 산정 교차검증 반복 수 (분할 잡음 완화)")
    ap.add_argument("--target-headroom", type=float, default=0.90,
                    help="out-of-fold 실현 비율이 한도의 이 비율을 넘지 않도록 마진을 정한다")
    ap.add_argument("--shift-worlds", type=int, default=60,
                    help="마진 산정 시 혼합비를 다시 뽑아 볼 가상 평가셋 수")
    ap.add_argument("--shift-concentration", type=float, default=2.0,
                    help="작을수록 혼합비가 크게 흔들린다. Train 계열 재추출만으로는 "
                         "Train→비공개셋 이동을 다 담지 못하므로 보수적으로 잡는다")
    ap.add_argument("--max-breach", type=float, default=0.02,
                    help="혼합비 변화 하에서 허용할 예산 초과 확률 상한")
    ap.add_argument("--predictor", choices=("2pl", "grm"), default="2pl",
                    help="grm 은 측정 후 기각된 대안이다 (fit_grm_items docstring 참조)")
    args = ap.parse_args()
    use_grm = args.predictor == "grm"

    policy = load_bundled_policy()
    rows = load_pairs(args.train_input, args.train_outcomes)
    texts = [r[0] for r in rows]
    n = len(texts)
    print(f"train episodes = {n}")

    Y = np.asarray([[float(o.score) for o in r[1]] for r in rows])
    TIN = np.asarray([[float(o.input_tokens) for o in r[1]] for r in rows])
    TOUT = np.asarray([[float(o.output_tokens) for o in r[1]] for r in rows])
    TRUE_C = np.asarray(
        [[cost_of(policy, MODELS[j], TIN[i, j], TOUT[i, j]) for j in range(3)]
         for i in range(n)]
    )

    X = design(texts, args.dim)
    print(f"design matrix = {X.shape}")

    # ---- out-of-fold predictions, used for margin sizing only ----------------
    #
    # Fold count matters more here than it looks. The shipped predictor is fitted
    # on all of Train; a fold model fitted on 80% is measurably noisier, and a
    # noisier score model buys worse upgrades. Because axk1-think costs ~22x the
    # light baseline, those bad buys are expensive: at the same margin, 5-fold
    # out-of-fold predictions spent ratio 3.29 on Train where the shipped model
    # spent 2.37 on the same episodes. Sizing the margin against that gap left
    # Premium using 2.66 of a 4.0 budget. More folds move the calibration model
    # closer to what actually ships; repeats average out the partition draw.
    # Each repeat is kept as its own prediction set. Averaging them would
    # denoise the predictor beyond anything that ships, and the whole point of
    # this step is to size a margin against the noise level that actually
    # deploys -- an averaged set reported a 0% breach rate for margins the
    # independent harness measured at 10-12%.
    reps = max(1, args.margin_repeats)
    oof_sets = [
        _oof_predictions(X, Y, TIN, TOUT, texts, policy, args,
                         args.margin_folds, seed=rep, use_grm=use_grm)
        for rep in range(reps)
    ]
    P_oof, C_oof = oof_sets[0]
    li = MODELS.index(policy.light_model_id)
    true_base = float(TRUE_C[:, li].sum())
    print("\nout-of-fold cost model check (before calibration)")
    cost_calib = []
    for j, m in enumerate(MODELS):
        ratio = float(C_oof[:, j].sum() / TRUE_C[:, j].sum())
        cost_calib.append(1.0 / ratio if ratio > 0 else 1.0)
        print(f"  {m:11s} predicted/actual = {ratio:.4f}  -> calibration x{cost_calib[-1]:.4f}")
    # Apply the correction to every out-of-fold cost set too, so the margin
    # sweep below sees the same cost scale the runtime will use.
    calib_arr = np.asarray(cost_calib)[None, :]
    calib_ones = np.ones((1, len(MODELS)))
    C_oof = C_oof * calib_arr

    # ---- size each tier's margin on out-of-fold realised ratio ---------------
    groups = _template_groups(texts)
    print(f"\ntemplate groups for shift resampling = {len(groups)}")
    tier_excess: Dict[str, float] = {}
    margin_report = {}
    for tier in TIERS:
        limit_mult = float(policy.tiers[tier].budget_multiplier)
        target = limit_mult * args.target_headroom
        chosen, rows_t = 0.05, []
        for cand in [x / 100.0 for x in range(5, 101, 5)]:
            picks = route(texts, P_oof, C_oof, tier, policy, cand)
            realised = sum(TRUE_C[i, MODELS.index(picks[i])] for i in range(n)) / true_base
            quality = float(np.mean([Y[i, MODELS.index(picks[i])] for i in range(n)]))
            # worst repeat, not the mean: a margin is only safe if it holds for
            # every prediction set we drew, not on average across them
            breach = max(
                breach_rate_under_shift(
                    texts, Pr * calib_ones, Cr * calib_arr, TRUE_C, policy, tier,
                    cand, groups, worlds=args.shift_worlds,
                    concentration=args.shift_concentration, seed=17 + r,
                )
                for r, (Pr, Cr) in enumerate(oof_sets)
            )
            rows_t.append({"excess": cand, "realised_ratio": realised,
                           "oof_quality": quality, "shift_breach": breach})
            # Two conditions, both on Train only. The point estimate keeps the
            # margin under the limit on the public mixture; the shift breach
            # rate keeps it under the limit on mixtures nobody has shown us.
            # Take the largest margin whose whole prefix passes: scanning for
            # the maximum passing candidate would step over a failing region
            # and ship a margin that is only safe by accident.
            if realised > target or breach > args.max_breach:
                break
            chosen = cand
        tier_excess[tier] = chosen
        best = [r for r in rows_t if r["excess"] == chosen][0]
        margin_report[tier] = {"chosen_excess": chosen, "target_ratio": target, **best,
                               "sweep": rows_t}
        print(f"  {tier:9s} limit {limit_mult:.2f}  target <= {target:.3f}  "
              f"chosen excess {chosen:.2f}  oof realised {best['realised_ratio']:.3f}  "
              f"shift breach {best['shift_breach']:.1%}  "
              f"oof quality {best['oof_quality']:.4f}")

    # ---- final fit on all of Train ------------------------------------------
    Win, sin_ = fit_tokens(X, TIN, args.alpha)
    Wout, sout = fit_tokens(X, TOUT, args.alpha)
    values = np.unique(Y)
    if use_grm:
        a, thr = fit_grm_items(Y, values)
        Wmu, Ws = fit_grm_heads(X, Y, a, thr, values)
        b = np.zeros(len(MODELS))
        print(f"\ngraded levels              = {values.tolist()}")
        print(f"final GRM discrimination a = {np.round(a, 4).tolist()}")
        for j, m in enumerate(MODELS):
            print(f"  {m:11s} thresholds = {np.round(thr[j], 4).tolist()}")
    else:
        Wmu, a, b = fit_2pl(X, Y, l2=args.alpha)
        Ws = np.zeros_like(Wmu)
        thr = np.zeros((len(MODELS), 0))
        print(f"\nfinal 2PL discrimination a = {np.round(a, 4).tolist()}")
        print(f"final 2PL difficulty     b = {np.round(b, 4).tolist()}")

    blob = {
        "schema_version": 1,
        "artifact_id": "lpb-router-v1",
        "models": list(MODELS),
        "dim": args.dim,
        "n_dense": N_DENSE,
        "grm": bool(use_grm),
        "values": [float(v) for v in values],
        "thresholds": [[float(v) for v in row] for row in thr],
        # featurize() already carries a constant column, so the head vectors
        # are exactly dim + N_DENSE long and need no appended intercept.
        "theta_w": [float(v) for v in Wmu],
        "sigma_w": [float(v) for v in Ws],
        "disc": [float(v) for v in a],
        "diff": [float(v) for v in b],
        "in_w": [[float(v) for v in row] for row in Win],
        "out_w": [[float(v) for v in row] for row in Wout],
        "in_smear": [float(v) for v in sin_],
        "out_smear": [float(v) for v in sout],
        "in_range": [[lo, hi] for lo, hi in token_ranges(TIN)],
        "out_range": [[lo, hi] for lo, hi in token_ranges(TOUT)],
        "cost_calib": [float(v) for v in cost_calib],
        "tier_excess": tier_excess,
        "train_episodes": n,
        "notes": "fitted from public Train only; margins sized on out-of-fold realised ratio",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(blob, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    size = args.artifact.stat().st_size
    print(f"\nwrote {args.artifact} ({size/1024:.0f} KiB)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"margins": margin_report,
                        "oof_cost_ratio": {m: float(C_oof[:, j].sum() / TRUE_C[:, j].sum())
                                           for j, m in enumerate(MODELS)},
                        "graded_levels": [float(v) for v in values],
                        "disc": [float(v) for v in a],
                        "thresholds": [[float(v) for v in row] for row in thr]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

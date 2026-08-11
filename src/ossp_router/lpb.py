# SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
# SPDX-License-Identifier: Apache-2.0

"""Prompt-only budget-allocating router (``router-run`` entry point).

The router reads prompts and a tier, and writes one v1 submission. It never
sees outcomes, ids-as-signal, or input order.

Design, in the order the numbers matter:

1. **Cost is the binding constraint, not difficulty.** ``axk1-think`` costs
   ~22x the all-light baseline on public Dev, so Fast (1.25x) can afford to
   escalate roughly 1% of episodes. A tier that overruns its limit by any
   amount scores exactly zero, so the cost model has to be right before the
   quality model matters at all.
2. **Predict tokens, not credits.** Input and output token counts are
   regressed separately in log space, then combined with the *known* published
   rates. Duan's smearing corrects the log-to-linear retransformation bias:
   ``axk1-think`` output tokens have mean 3185 against median 1499, so
   ``exp(mean of log)`` understates the arithmetic mean the budget is charged.
3. **Allocate over the whole batch.** ``docs/RUNTIME.md`` delivers the entire
   input in one run, and the budget is a single total over that batch. Each
   episode contributes a concave quality/cost envelope; upgrades are bought in
   decreasing order of predicted quality gain per credit until the planned
   spend reaches the tier's calibrated ceiling.
4. **Spend the margin where it is cheap.** The safety margin applies to the
   *excess* over all-light, never to the whole limit: all-light is 1.0x by
   definition and cannot breach. Per-tier margins are fitted offline from
   out-of-fold realised ratios, not from a public-Dev score.

Everything learned lives in ``resources/lpb-artifact.v1.json``, produced by
``tools/train_lpb.py`` from public Train only. This module is standard library
only so the submission image needs no third-party runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import zlib
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .protocol import (
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_policy,
    loads_json,
    parse_submission,
    submission_to_dict,
)
from .heuristic import (
    episode_text,
    extract_features,
    select_model,
    write_submission_atomic,
)

ARTIFACT_NAME = "lpb-artifact.v1.json"

#: Token budget per prompt for feature extraction. BABILong prompts reach
#: 65k characters; hashing all of it would put the 90 s per-tier limit at risk
#: for no routing signal. Head and tail are kept because long-context items
#: carry their question at the end.
_HEAD_TOKENS = 420
_TAIL_TOKENS = 92

_WORD = re.compile(r"[a-z0-9]+|[가-힣]+")
_CODE = re.compile(r"def \w+\(|assert |return |import |```|#include|SELECT ", re.I)
_MCQ = re.compile(r"\n\s*[A-D][.)]\s")
_MATH = re.compile(r"[=+\-*/^<>]|\\(?:frac|sum|int|sqrt)\b")
_REASON = re.compile(r"\b(prove|derive|explain|why|analyz|algorithm|complexity)\b", re.I)
_COUNTING = re.compile(r"\b(how many|how much|total|calculate|sum of)\b", re.I)
_SENT = re.compile(r"[.!?。！？]")

N_DENSE = 16


def _tokens(text: str) -> List[str]:
    toks = _WORD.findall(text.lower())
    if len(toks) <= _HEAD_TOKENS + _TAIL_TOKENS:
        return toks
    return toks[:_HEAD_TOKENS] + toks[-_TAIL_TOKENS:]


def featurize(text: str, dim: int) -> List[float]:
    """Signed feature hashing over unigrams and bigrams, plus dense features.

    ``zlib.crc32`` is used rather than ``hash()`` because ``hash()`` is salted
    per process: identical inputs would route differently between runs, which
    the operator's repeat-run audit would catch.
    """
    vec = [0.0] * (dim + N_DENSE)
    toks = _tokens(text)
    n = len(toks)
    for i, tok in enumerate(toks):
        h = zlib.crc32(tok.encode("utf-8"))
        vec[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
        if i + 1 < n:
            h2 = zlib.crc32(b"\x1f".join((tok.encode("utf-8"), toks[i + 1].encode("utf-8"))))
            vec[h2 % dim] += 1.0 if (h2 >> 31) & 1 else -1.0
    if n:
        scale = 1.0 / math.sqrt(n)
        for i in range(dim):
            vec[i] *= scale

    chars = len(text)
    nonspace = sum(1 for c in text if not c.isspace()) or 1
    hangul = sum(1 for c in text if "가" <= c <= "힣")
    digits = sum(1 for c in text if c.isdigit())
    d = dim
    vec[d + 0] = math.log1p(chars)
    vec[d + 1] = math.log1p(n)
    vec[d + 2] = digits / nonspace
    vec[d + 3] = hangul / nonspace
    vec[d + 4] = 1.0 if _CODE.search(text) else 0.0
    vec[d + 5] = len(_MATH.findall(text)) / nonspace
    vec[d + 6] = 1.0 if chars > 4000 else 0.0
    vec[d + 7] = 1.0 if _MCQ.search(text) else 0.0
    vec[d + 8] = text.count("\n") / nonspace * 100.0
    vec[d + 9] = 1.0 if _REASON.search(text) else 0.0
    vec[d + 10] = 1.0 if _COUNTING.search(text) else 0.0
    vec[d + 11] = len(set(toks)) / max(1, n)
    vec[d + 12] = math.log1p(len(_SENT.findall(text)))
    vec[d + 13] = 1.0 if text.lstrip().startswith("Question:") else 0.0
    vec[d + 14] = math.log1p(chars / max(1, n))
    vec[d + 15] = 1.0
    return vec


def content_key(text: str) -> int:
    """Deterministic content-only tie-break. Never an id, never a position."""
    return zlib.crc32(text.encode("utf-8"))


def _dot(w: Sequence[float], x: Sequence[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    e = math.exp(max(z, -60.0))
    return e / (1.0 + e)


class Artifact:
    """Learned parameters, fitted offline on public Train only."""

    def __init__(self, blob: dict):
        self.dim = int(blob["dim"])
        self.models: List[str] = list(blob["models"])
        self.theta_w: List[float] = [float(v) for v in blob["theta_w"]]
        self.disc: List[float] = [float(v) for v in blob["disc"]]
        self.diff: List[float] = [float(v) for v in blob["diff"]]
        self.in_w: List[List[float]] = [[float(v) for v in row] for row in blob["in_w"]]
        self.out_w: List[List[float]] = [[float(v) for v in row] for row in blob["out_w"]]
        self.in_smear: List[float] = [float(v) for v in blob["in_smear"]]
        self.out_smear: List[float] = [float(v) for v in blob["out_smear"]]
        #: effective fraction of each tier's *excess* budget the router may plan
        #: to spend, chosen offline from out-of-fold realised ratios
        self.tier_excess: Dict[str, float] = {
            str(k): float(v) for k, v in blob["tier_excess"].items()
        }
        expected = self.dim + N_DENSE
        for name, vec in (("theta_w", self.theta_w),):
            if len(vec) != expected:
                raise ProtocolError(f"{name} 길이가 {expected}이어야 합니다: {len(vec)}")
        if len(self.models) != len(self.disc) != len(self.diff):
            raise ProtocolError("모델 수와 IRT 파라미터 수가 다릅니다.")

    def predict(self, feats: Sequence[float]) -> Tuple[List[float], List[float]]:
        """Return (per-model score, per-model cost-in-credits) for one prompt."""
        theta = _dot(self.theta_w, feats)
        scores = [
            _sigmoid(a * (theta - b)) for a, b in zip(self.disc, self.diff)
        ]
        costs = []
        for j in range(len(self.models)):
            tin = math.exp(min(_dot(self.in_w[j], feats), 20.0)) * self.in_smear[j]
            tout = math.exp(min(_dot(self.out_w[j], feats), 20.0)) * self.out_smear[j]
            costs.append((tin, tout))
        return scores, costs


def load_artifact(path: Optional[Path] = None) -> Optional[Artifact]:
    """Load the learned artifact, or ``None`` if it is absent or unreadable.

    A missing artifact is not fatal: :func:`make_submission` falls back to the
    bundled heuristic so the container still emits a valid submission rather
    than failing the tier outright.
    """
    try:
        if path is not None:
            text = Path(path).read_text(encoding="utf-8")
        else:
            text = resources.read_text(
                "ossp_router.resources", ARTIFACT_NAME, encoding="utf-8"
            )
        return Artifact(loads_json(text))
    except (OSError, UnicodeError, ProtocolError, ValueError, KeyError, TypeError):
        return None


def _episode_cost(policy: RoutingPolicy, model_id: str, tin: float, tout: float) -> float:
    rates = policy.models[model_id]
    unit = float(policy.token_unit)
    return (
        float(rates.fixed_cost)
        + tin * float(rates.input_token_rate) / unit
        + tout * float(rates.output_token_rate) / unit
    )


def _envelope(costs: Sequence[float], scores: Sequence[float], names: Sequence[str]):
    """Upper concave envelope of (cost, score) options, cheapest first."""
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


def allocate(
    texts: Sequence[str],
    tier: str,
    policy: RoutingPolicy,
    artifact: Artifact,
) -> List[str]:
    """Choose one model per prompt under the tier's global cost ceiling."""
    light = policy.light_model_id
    models = artifact.models
    n = len(texts)
    if n == 0:
        return []

    plans = []
    base_pred = 0.0
    for text in texts:
        feats = featurize(text, artifact.dim)
        scores, tokens = artifact.predict(feats)
        costs = [
            _episode_cost(policy, models[j], tokens[j][0], tokens[j][1])
            for j in range(len(models))
        ]
        base_pred += costs[models.index(light)]
        plans.append(_envelope(costs, scores, models))

    multiplier = float(policy.tiers[tier].budget_multiplier)
    excess = artifact.tier_excess.get(tier, 0.30)
    limit = base_pred * (1.0 + (multiplier - 1.0) * excess)

    upgrades = []
    for i, hull in enumerate(plans):
        key = content_key(texts[i])
        for k in range(1, len(hull)):
            dc = hull[k][0] - hull[k - 1][0]
            ds = hull[k][1] - hull[k - 1][1]
            if dc > 0.0 and ds > 0.0:
                upgrades.append((-ds / dc, key, k, i, dc))
    upgrades.sort()

    spent = sum(hull[0][0] for hull in plans)
    level = [0] * n
    for _, _, k, i, dc in upgrades:
        if level[i] != k - 1:
            continue
        if spent + dc <= limit:
            spent += dc
            level[i] = k
    return [plans[i][level[i]][2] for i in range(n)]


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
    *,
    artifact: Optional[Artifact] = None,
) -> Submission:
    """Create one complete v1 submission for a single tier."""
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")

    if artifact is None:
        artifact = load_artifact()
    if artifact is None:
        model_ids = [
            select_model(extract_features(ep), tier) for ep in inputs.episodes
        ]
    else:
        texts = [episode_text(ep) for ep in inputs.episodes]
        model_ids = allocate(texts, tier, policy, artifact)

    decisions = tuple(
        Decision(ep.episode_id, model_id)
        for ep, model_id in zip(inputs.episodes, model_ids)
    )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=decisions,
    )
    # Keep the generator and the public v1 parser on the same strict path.
    return parse_submission(submission_to_dict(submission))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="프롬프트 기반 예산 배분 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifact", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        artifact = load_artifact(args.artifact)
        submission = make_submission(inputs, policy, args.tier, artifact=artifact)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

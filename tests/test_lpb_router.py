# SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
# SPDX-License-Identifier: Apache-2.0

"""Rule-compliance tests for the LPB router.

The operator reserves the right to re-run a submission with shuffled episode
ids and input order (``docs/CHALLENGE_RULES.md``), and participants cannot opt
out of that audit. These tests run it locally first.

Batch-level allocation is legal and, given a single total cost constraint, close
to necessary, so a prompt's selection may legitimately move when the surrounding
batch changes. What must never move is the selection under a change of id or of
position alone.
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import lpb  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    Episode,
    InputBatch,
    load_bundled_policy,
    load_input,
    parse_input,
    submission_to_dict,
)

TOY = ROOT / "data" / "toy" / "inputs.json"
DEV = ROOT / "data" / "materialized" / "dev" / "inputs.json"


def _decisions(submission):
    return {d.episode_id: d.model_id for d in submission.decisions}


class LpbRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()
        cls.artifact = lpb.load_artifact()
        cls.inputs = load_input(DEV if DEV.exists() else TOY)

    def _run(self, inputs, tier):
        return _decisions(
            lpb.make_submission(inputs, self.policy, tier, artifact=self.artifact)
        )

    def test_artifact_is_bundled(self):
        self.assertIsNotNone(
            self.artifact, "resources/lpb-artifact.v1.json 을 읽지 못했습니다"
        )
        self.assertEqual(list(self.artifact.models), list(self.policy.models))
        for tier in TIERS:
            self.assertIn(tier, self.artifact.tier_excess)

    def test_every_episode_decided_once_with_known_model(self):
        for tier in TIERS:
            got = self._run(self.inputs, tier)
            self.assertEqual(
                set(got), {e.episode_id for e in self.inputs.episodes}
            )
            for model_id in got.values():
                self.assertIn(model_id, self.policy.models)

    def test_deterministic_across_runs(self):
        for tier in TIERS:
            first = self._run(self.inputs, tier)
            for _ in range(2):
                self.assertEqual(first, self._run(self.inputs, tier))

    def test_episode_id_has_no_influence(self):
        rng = random.Random(20260811)
        tokens = [f"audit-{rng.getrandbits(48):012x}" for _ in self.inputs.episodes]
        renamed = InputBatch(
            schema_version=self.inputs.schema_version,
            challenge_id=self.inputs.challenge_id,
            split=self.inputs.split,
            episodes=tuple(
                Episode(tok, ep.prompt, ep.messages)
                for tok, ep in zip(tokens, self.inputs.episodes)
            ),
        )
        for tier in TIERS:
            base = self._run(self.inputs, tier)
            other = self._run(renamed, tier)
            for ep, tok in zip(self.inputs.episodes, tokens):
                self.assertEqual(
                    base[ep.episode_id], other[tok], f"{tier}: {ep.episode_id}"
                )

    def test_input_order_has_no_influence(self):
        shuffled = list(self.inputs.episodes)
        random.Random(7).shuffle(shuffled)
        reordered = InputBatch(
            schema_version=self.inputs.schema_version,
            challenge_id=self.inputs.challenge_id,
            split=self.inputs.split,
            episodes=tuple(shuffled),
        )
        for tier in TIERS:
            base = self._run(self.inputs, tier)
            other = self._run(reordered, tier)
            self.assertEqual(base, other, f"{tier}: 입력 순서가 선택을 바꿨습니다")

    def test_feature_hash_is_not_process_salted(self):
        """``hash()`` is salted per process; the router must not depend on it."""
        text = "Question: what is 2 + 2?\nA. 4\nB. 5"
        self.assertEqual(lpb.content_key(text), lpb.content_key(text))
        self.assertEqual(
            lpb.featurize(text, 64), lpb.featurize(text, 64)
        )

    def test_tier_spend_is_monotone(self):
        """A richer tier must not plan to spend less than a poorer one."""
        order = {"ax31-light": 0, "ax31": 1, "axk1-think": 2}
        totals = []
        for tier in TIERS:
            got = self._run(self.inputs, tier)
            totals.append(sum(order[m] for m in got.values()))
        self.assertLessEqual(totals[0], totals[1])
        self.assertLessEqual(totals[1], totals[2])

    def test_falls_back_to_heuristic_without_artifact(self):
        """A missing artifact must still yield a valid submission, not a failure."""
        small = parse_input(
            {
                "schema_version": self.inputs.schema_version,
                "challenge_id": self.inputs.challenge_id,
                "split": self.inputs.split,
                "episodes": [
                    {"episode_id": e.episode_id, "prompt": lpb.episode_text(e)}
                    for e in self.inputs.episodes[:20]
                ],
            }
        )
        sub = lpb.make_submission(small, self.policy, "fast", artifact=None)
        payload = submission_to_dict(sub)
        self.assertEqual(len(payload["decisions"]), 20)

    def test_empty_batch_is_handled(self):
        self.assertEqual(lpb.allocate([], "fast", self.policy, self.artifact), [])


if __name__ == "__main__":
    unittest.main()

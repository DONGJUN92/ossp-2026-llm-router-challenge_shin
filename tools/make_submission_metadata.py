# SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
# SPDX-License-Identifier: Apache-2.0

"""Write ``submission-ossp-skt.json`` from the repository's current state.

    python3 tools/make_submission_metadata.py \
        --image-digest ghcr.io/dongjun92/ossp-router@sha256:<64 hex>

``docs/SUBMISSION.md`` requires the JSON to name a code commit that already
exists and an image digest built from *that* commit, and the JSON itself must
land in a later, separate commit. This script therefore refuses to guess: it
reads the current HEAD, checks the tree is clean, and takes the digest from the
command line, because a registry digest only exists after a push and cannot be
derived locally.

Order of operations, from ``docs/SUBMISSION.md``:

1. commit the final router code and note that SHA
2. build ``linux/arm64`` from that commit and push it
3. run this script with the pushed digest, which writes the JSON
4. commit the JSON on its own
5. put the step-4 commit's snapshot URL in the report
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission-ossp-skt.json"
CHALLENGE_ID = "ossp-2026-llm-router-challenge"
DIGEST_RE = re.compile(
    r"^(?:(?:localhost|[a-z0-9]+(?:[.-][a-z0-9]+)*)(?::[0-9]+)?/)?"
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*@sha256:[0-9a-f]{64}$"
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-digest", required=True,
                    help="푸시한 이미지의 저장소명@sha256:... 전체 다이제스트")
    ap.add_argument("--repository-url",
                    help="공개 저장소 기본 URL. 생략하면 origin 에서 유추한다")
    ap.add_argument("--commit-sha",
                    help="이미지를 빌드한 코드 커밋. 생략하면 현재 HEAD")
    ap.add_argument("--primary-license", default="Apache-2.0")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if not DIGEST_RE.match(args.image_digest):
        print(f"오류: image_digest 형식이 스키마와 다릅니다: {args.image_digest}",
              file=sys.stderr)
        return 2

    if not args.allow_dirty and _git("status", "--porcelain"):
        print("오류: 작업 트리가 깨끗하지 않습니다. 코드 커밋을 먼저 확정하십시오.",
              file=sys.stderr)
        return 2

    commit = args.commit_sha or _git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        print(f"오류: commit_sha 는 40자리 소문자 16진수여야 합니다: {commit}",
              file=sys.stderr)
        return 2

    url = args.repository_url
    if not url:
        remote = _git("remote", "get-url", "origin")
        url = remote[:-4] if remote.endswith(".git") else remote
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):]
    if not url.startswith("https://"):
        print(f"오류: repository_url 은 https 여야 합니다: {url}", file=sys.stderr)
        return 2

    payload = {
        "schema_version": 1,
        "challenge_id": CHALLENGE_ID,
        "repository_url": url,
        "commit_sha": commit,
        "image_digest": args.image_digest,
        "primary_license": args.primary_license,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("\n이 파일만 담은 커밋을 따로 만들고, 그 커밋의 스냅샷 URL을 "
          "결과보고서의 '프로젝트 등록 URL'에 적으십시오:")
    print(f"  {url}/tree/<이 JSON 을 담은 커밋 SHA>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

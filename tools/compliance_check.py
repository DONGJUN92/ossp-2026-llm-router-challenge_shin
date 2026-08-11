# SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
# SPDX-License-Identifier: Apache-2.0

"""Mechanical compliance harness for every checkable rule in the seven SKT docs.

    python3 tools/compliance_check.py --image ghcr.io/<ns>/<name>:submission

Groups map to the documents:

  A  submission metadata and repository       SUBMISSION.md, CHALLENGE_RULES.md
  B  container image properties               RUNTIME.md, ENFORCEMENT.md
  C  runtime behaviour under isolation        RUNTIME.md
  D  rule compliance and audit simulation     CHALLENGE_RULES.md, ENFORCEMENT.md
  E  licensing and bundled-file provenance    CHALLENGE_RULES.md, ENFORCEMENT.md
  F  data hygiene                             DATA_CARD.md, DATA_LICENSES.md

Every check prints PASS, FAIL or SKIP with the rule it enforces. A FAIL in group
D or E is a disqualification risk, not a lost tier; those are reported first in
the summary. Checks needing Docker are skipped with a reason when it is absent
rather than silently passing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHALLENGE_ID = "ossp-2026-llm-router-challenge"
POLICY_ID = "ossp-2026-prompt-router-v1"
MODELS = {"ax31-light", "ax31", "axk1-think"}
TIERS = ("fast", "balanced", "premium")
ALLOWED_LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD"}
UPSTREAM_RELEASE = "3cccbf602077a846c13b2cb1356eee1559a631db"

RESULTS: list[tuple[str, str, str, str]] = []   # (group, id, status, detail)
VERIFY_REBUILD = False


def rec(group: str, cid: str, ok, detail: str, rule: str = "") -> None:
    status = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    RESULTS.append((group, cid, status, detail))
    mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    line = f"  [{mark}] {cid:6s} {detail}"
    if rule and status != "PASS":
        line += f"\n           규칙: {rule}"
    print(line, flush=True)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def sh(args: list[str], timeout: int = 600):
    # explicit utf-8: the router's messages are Korean and the Windows console
    # codec (cp949) raises UnicodeDecodeError inside subprocess's reader thread,
    # which silently turns stdout/stderr into None and hides the real failure.
    r = subprocess.run(args, capture_output=True, timeout=timeout)
    out = (r.stdout or b"").decode("utf-8", "replace")
    err = (r.stderr or b"").decode("utf-8", "replace")
    return subprocess.CompletedProcess(r.args, r.returncode, out, err)


# ---------------------------------------------------------------- A metadata
def group_a() -> dict:
    print("\nA. 제출 정보와 저장소  (SUBMISSION.md / CHALLENGE_RULES.md)")
    meta = {}
    p = ROOT / "submission-ossp-skt.json"
    if not p.exists():
        rec("A", "A1", False, "submission-ossp-skt.json 이 저장소 루트에 없음",
            "SUBMISSION.md: 제출 저장소 루트에 반드시 커밋")
        return meta
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        rec("A", "A1", False, f"JSON 파싱 실패: {exc}")
        return meta
    allowed = {"schema_version", "challenge_id", "repository_url",
               "commit_sha", "image_digest", "primary_license"}
    extra = set(meta) - allowed
    missing = allowed - set(meta)
    rec("A", "A1", not extra and not missing,
        f"6개 필드 정확히 존재 (초과={sorted(extra)} 누락={sorted(missing)})",
        "SUBMISSION.md: 다음 여섯 필드만 허용")
    rec("A", "A2", meta.get("schema_version") == 1,
        f"schema_version == 1 (실제 {meta.get('schema_version')!r})")
    rec("A", "A3", meta.get("challenge_id") == CHALLENGE_ID,
        f"challenge_id 일치 ({meta.get('challenge_id')!r})")
    rec("A", "A4", meta.get("primary_license") in ALLOWED_LICENSES,
        f"primary_license 허용 목록 ({meta.get('primary_license')!r})",
        "CHALLENGE_RULES.md: Apache-2.0, MIT, BSD-2/3-Clause, ISC, 0BSD")

    sha = str(meta.get("commit_sha", ""))
    rec("A", "A5", bool(re.fullmatch(r"[0-9a-f]{40}", sha)),
        f"commit_sha 40자리 소문자 16진수 ({sha[:12]}...)")
    exists = git("cat-file", "-t", sha) == "commit" if sha else False
    rec("A", "A6", exists, f"commit_sha 가 이 저장소에 실제로 존재")
    if exists:
        anc = subprocess.run(["git", "-C", str(ROOT), "merge-base",
                              "--is-ancestor", sha, "HEAD"]).returncode == 0
        rec("A", "A7", anc or sha == git("rev-parse", "HEAD"),
            "commit_sha 가 HEAD 의 조상이거나 HEAD 자신",
            "SUBMISSION.md: JSON 은 코드 커밋보다 뒤 커밋에 담긴다")

    dig = str(meta.get("image_digest", ""))
    pat = (r"^(?:(?:localhost|[a-z0-9]+(?:[.-][a-z0-9]+)*)(?::[0-9]+)?/)?"
           r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
           r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*@sha256:[0-9a-f]{64}$")
    rec("A", "A8", bool(re.match(pat, dig)), f"image_digest 스키마 형식 ({dig[:48]}...)")
    url = str(meta.get("repository_url", ""))
    rec("A", "A9", url.startswith("https://") and "/tree/" not in url and "/commit/" not in url,
        f"repository_url https 이고 커밋 경로 없음 ({url})")

    # fork lineage
    is_fork = subprocess.run(["git", "-C", str(ROOT), "merge-base",
                              "--is-ancestor", UPSTREAM_RELEASE, "HEAD"]).returncode == 0
    rec("A", "A10", is_fork, f"공식 릴리스 커밋 {UPSTREAM_RELEASE[:8]} 을 조상으로 포함 (fork)",
        "SUBMISSION.md: 이 저장소를 fork 한 공개 저장소")
    rec("A", "A11", not (ROOT / ".gitmodules").exists(),
        "git submodule 없음", "CHALLENGE_RULES.md: 비공개 submodule 의존 금지")
    rec("A", "A12", git("status", "--porcelain") == "",
        "작업 트리 clean (제출 소스와 빌드 내용 일치의 전제)")
    lic = (ROOT / "LICENSE")
    rec("A", "A13", lic.exists() and "Apache License" in lic.read_text(encoding="utf-8")[:2000],
        "루트 LICENSE 가 Apache-2.0")
    return meta


# ---------------------------------------------------------------- B image
def group_b(image: str | None, meta: dict) -> bool:
    print("\nB. 컨테이너 이미지  (RUNTIME.md / ENFORCEMENT.md)")
    if not shutil.which("docker"):
        rec("B", "B*", None, "docker CLI 없음 — 이미지 검사 생략")
        return False
    if sh(["docker", "version", "--format", "{{.Server.Arch}}"]).returncode != 0:
        rec("B", "B*", None, "docker 데몬 미동작 — 이미지 검사 생략")
        return False
    if not image:
        rec("B", "B*", None, "--image 미지정 — 이미지 검사 생략")
        return False
    # full JSON rather than a --format template: {{json .Config.Volumes}} errors
    # out when the key is absent, which is exactly the compliant case.
    ins = sh(["docker", "image", "inspect", image])
    if ins.returncode != 0:
        rec("B", "B1", False, f"이미지를 찾을 수 없음: {image} ({ins.stderr.strip()[:120]})")
        return False
    info = json.loads(ins.stdout)[0]
    plat = f"{info.get('Os')}/{info.get('Architecture')}"
    iid = info.get("Id", "")
    vols = info.get("Config", {}).get("Volumes")
    vols = "null" if vols in (None, {}) else json.dumps(vols)
    size = info.get("Size", 0)
    rec("B", "B1", plat == "linux/arm64", f"플랫폼 linux/arm64 (실제 {plat})",
        "RUNTIME.md: linux/arm64 가 아닌 이미지는 받지 않습니다")
    rec("B", "B2", vols in ("null", "{}", ""), f"VOLUME 선언 없음 (Config.Volumes={vols})",
        "ENFORCEMENT.md: VOLUME 선언 이미지는 접수 거부")
    apparent = int(size)
    rec("B", "B3", apparent <= 2 * 1024 ** 3,
        f"병합 루트 파일시스템 {apparent/1024**2:.0f} MiB <= 2 GiB")
    layer_note = "레지스트리 조회 불가"
    layers_ok: bool | None = None
    dig = str(meta.get("image_digest", ""))
    if "@sha256:" in dig:
        want = dig.split("@")[1]
        rec("B", "B5", iid == want,
            f"로컬 이미지 ID 가 제출 다이제스트와 일치 ({iid[:19]} vs {want[:19]})",
            "ENFORCEMENT.md: 제출 커밋과 이미지 다이제스트의 대응")
        ref = dig.split("@")[0]
        ns = "/".join(ref.split("/")[1:]) if ref.startswith("ghcr.io/") else None
        if ns:
            tok = sh(["curl.exe", "-s",
                      f"https://ghcr.io/token?scope=repository:{ns}:pull&service=ghcr.io"])
            m = re.search(r'"token":"([^"]+)"', tok.stdout or "")
            if m:
                r = sh(["curl.exe", "-s", "-o", os.devnull, "-w", "%{http_code}",
                        "-H", f"Authorization: Bearer {m.group(1)}",
                        "-H", "Accept: application/vnd.oci.image.manifest.v1+json",
                        f"https://ghcr.io/v2/{ns}/manifests/{want}"])
                rec("B", "B6", r.stdout.strip() == "200",
                    f"익명(비로그인) 접근으로 다이제스트 조회 HTTP {r.stdout.strip()} — 공개 상태",
                    "CHALLENGE_RULES.md: 별도 권한 없이 공개 접근 가능해야 함")
                man = sh(["curl.exe", "-s", "-H", f"Authorization: Bearer {m.group(1)}",
                          "-H", "Accept: application/vnd.oci.image.manifest.v1+json",
                          f"https://ghcr.io/v2/{ns}/manifests/{want}"])
                try:
                    mj = json.loads(man.stdout)
                    tot = sum(int(l["size"]) for l in mj.get("layers", []))
                    layers_ok = tot <= 1024 ** 3
                    layer_note = (f"압축 계층 {len(mj['layers'])}개 합계 "
                                  f"{tot/1024**2:.1f} MiB <= 1 GiB")
                except Exception as exc:
                    layer_note = f"매니페스트 파싱 실패: {exc}"
            else:
                rec("B", "B6", None, "익명 토큰 발급 실패 — 공개 여부 미확인")
    rec("B", "B4", layers_ok, layer_note,
        "RUNTIME.md: OCI 압축 계층 합계 1 GiB")

    if VERIFY_REBUILD:
        # The strongest available evidence for two separate rules: the image can
        # be rebuilt from the submitted commit, and the commit corresponds to the
        # immutable digest. Requires a clean tree to mean anything.
        tag = "ossp-compliance-rebuild:check"
        b = sh(["docker", "build", "--pull", "--platform", "linux/arm64",
                "--provenance=false", "--sbom=false", "--file",
                str(ROOT / "container" / "Dockerfile"), "--tag", tag, str(ROOT)],
               timeout=1800)
        if b.returncode != 0:
            rec("B", "B7", False, f"재빌드 실패: {b.stderr.strip()[-200:]}")
        else:
            got = sh(["docker", "image", "inspect", tag, "--format", "{{.Id}}"]).stdout.strip()
            want = dig.split("@")[1] if "@sha256:" in dig else ""
            rec("B", "B7", got == want,
                f"현재 커밋에서 재빌드한 다이제스트가 제출본과 일치 "
                f"({got[:19]} vs {want[:19]})",
                "CHALLENGE_RULES.md: 제출한 커밋에서 이미지를 재현 가능하게 빌드")
            sh(["docker", "image", "rm", "-f", tag])
    else:
        rec("B", "B7", None, "재빌드 검증 생략 (--verify-rebuild 로 활성화)")
    return True


# ---------------------------------------------------------------- E licensing
def group_e() -> None:
    print("\nE. 라이선스와 포함 파일 고지  (CHALLENGE_RULES.md / ENFORCEMENT.md)")
    art = ROOT / "src" / "ossp_router" / "resources" / "lpb-artifact.v1.json"
    rec("E", "E1", art.exists(), f"학습 산출물 존재 ({art.name})")
    doc = ROOT / "docs" / "ROUTER.md"
    txt = doc.read_text(encoding="utf-8") if doc.exists() else ""
    if art.exists():
        import hashlib
        h = hashlib.sha256(art.read_bytes()).hexdigest()
        rec("E", "E2", h in txt,
            f"산출물 SHA-256 이 docs/ROUTER.md 에 기재 ({h[:16]}...)",
            "SUBMISSION.md: 포함 파일의 SHA-256 기록")
        rec("E", "E3", "Apache-2.0" in txt and "NumPy" in txt,
            "산출물 라이선스와 학습 의존성이 문서화")
    # no third-party weights in the image path
    src = ROOT / "src" / "ossp_router"
    big = [p for p in src.rglob("*") if p.is_file() and p.stat().st_size > 5 * 1024 ** 2]
    rec("E", "E4", not big, f"이미지 경로에 5 MiB 초과 파일 없음 (제3자 가중치 미포함) {[p.name for p in big]}")
    rt = (src / "lpb.py").read_text(encoding="utf-8")
    forbidden = ["import numpy", "import torch", "import scipy", "sentence_transformers",
                 "import requests", "urllib", "socket", "http.client"]
    hits = [f for f in forbidden if f in rt]
    rec("E", "E5", not hits, f"런타임 모듈에 제3자·네트워크 import 없음 {hits}",
        "CHALLENGE_RULES.md: 네트워크·외부 추론 호출 금지")
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    rec("E", "E6", "!src/ossp_router/lpb.py" in di
        and "!src/ossp_router/resources/lpb-artifact.v1.json" in di,
        "라우터와 산출물이 .dockerignore 허용목록에 포함")


# ---------------------------------------------------------------- F data
def group_f() -> None:
    print("\nF. 데이터 위생  (DATA_CARD.md / DATA_LICENSES.md)")
    tracked = git("ls-files").splitlines()
    mat = [p for p in tracked if p.startswith("data/materialized/")]
    rec("F", "F1", not mat, f"data/materialized 가 추적되지 않음 {mat[:3]}",
        "DATA_CARD.md: AIME 원문·캐시를 저장소에 넣지 않는다")
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    rec("F", "F2", "/data/materialized/" in gi and "/data/cache/" in gi,
        ".gitignore 가 materialized·cache 를 제외")
    suspicious = [p for p in tracked
                  if re.search(r"(answer|gold|solution|generation|response)", p, re.I)]
    rec("F", "F3", not suspicious, f"정답·생성물로 보이는 추적 파일 없음 {suspicious[:3]}")
    big = [p for p in tracked
           if (ROOT / p).exists() and (ROOT / p).stat().st_size > 20 * 1024 ** 2]
    rec("F", "F4", not big, f"20 MiB 초과 추적 파일 없음 {big[:3]}")


# ---------------------------------------------------------------- C runtime
RUN_FLAGS = [
    "--rm", "--platform", "linux/arm64", "--network", "none", "--read-only",
    "--user", "65532:65532", "--cpus", "2", "--memory", "2g", "--memory-swap", "2g",
    "--pids-limit", "32", "--ipc", "none", "--cgroupns", "private",
    "--ulimit", "core=0:0", "--log-driver", "none", "--no-healthcheck",
    "--stop-signal", "SIGTERM", "--tmpfs", "/tmp:size=256m",
]


def _write_input(path: Path, episodes: list, challenge_id=CHALLENGE_ID, split="probe") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"schema_version": 1, "challenge_id": challenge_id, "split": split,
         "episodes": episodes}, ensure_ascii=False, separators=(",", ":"),
        sort_keys=True), encoding="utf-8")


def _prepare_out_volume(vol: str) -> None:
    """Fresh volume owned by the container's UID.

    A new named volume is root-owned 0755, so a container running as 65532
    cannot write and the router exits 2. RUNTIME.md says the operator supplies a
    writable restricted output volume, so the harness has to supply one too -
    otherwise this checks the harness, not the submission.
    """
    sh(["docker", "volume", "rm", "-f", vol])
    sh(["docker", "volume", "create", vol])
    sh(["docker", "run", "--rm", "-v", f"{vol}:/o", "--entrypoint", "sh",
        "busybox:1.36", "-c", "chown 65532:65532 /o && chmod 755 /o"])


def _run_container(image: str, in_dir: Path, out_vol: str, tier: str, timeout=180):
    """one container run using a named volume for output, so POSIX modes apply"""
    _prepare_out_volume(out_vol)
    args = ["docker", "run", *RUN_FLAGS,
            "-v", f"{in_dir}:/challenge/input:ro",
            "-v", f"{out_vol}:/challenge/output",
            image,
            "--input", "/challenge/input/inputs.json",
            "--tier", tier,
            "--output", "/challenge/output/submission.json"]
    import time
    t0 = time.monotonic()
    r = sh(args, timeout=timeout)
    return r, time.monotonic() - t0


def _read_out_volume(out_vol: str) -> tuple[str, str]:
    """listing (with modes) and the submission body, read from inside a container"""
    lst = sh(["docker", "run", "--rm", "-v", f"{out_vol}:/o", "--entrypoint", "sh",
              "busybox:1.36", "-c", "ls -laA /o"])
    body = sh(["docker", "run", "--rm", "-v", f"{out_vol}:/o", "--entrypoint", "sh",
               "busybox:1.36", "-c", "cat /o/submission.json 2>/dev/null || true"])
    return lst.stdout, body.stdout


def group_c(image: str | None, have_docker: bool) -> dict:
    print("\nC. 격리 조건에서의 실행  (RUNTIME.md)")
    outs = {}
    if not (have_docker and image):
        rec("C", "C*", None, "docker 또는 이미지 없음 — 실행 검사 생략")
        return outs
    from ossp_router.protocol import load_input, parse_submission

    inputs = load_input(ROOT / "data" / "materialized" / "train" / "inputs.json")
    dev = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    eps = [{"episode_id": e.episode_id,
            **({"prompt": e.prompt} if e.prompt is not None
               else {"messages": [{"role": m.role, "content": m.content} for m in e.messages]})}
           for e in list(inputs.episodes) + list(dev.episodes)]
    work = Path(tempfile.mkdtemp(prefix="ossp-compl-"))
    in_dir = work / "in"
    _write_input(in_dir / "inputs.json", eps, split="public-train-dev")
    print(f"       입력 {len(eps)}문항, {in_dir}")

    for tier in TIERS:
        vol = f"ossp-compl-{tier}"
        r, secs = _run_container(image, in_dir, vol, tier)
        rec("C", f"C1{tier[0]}", r.returncode == 0,
            f"{tier}: 종료코드 {r.returncode}"
            + (f" | stderr: {r.stderr.strip()[:300]}" if r.returncode else ""))
        rec("C", f"C2{tier[0]}", secs <= 90.0,
            f"{tier}: {secs:.1f}s <= 90s (에뮬레이션이라 실제 장비보다 느림)",
            "RUNTIME.md: 등급별 실행 시간 90초")
        rec("C", f"C3{tier[0]}", len(r.stdout.encode()) <= 1024 ** 2
            and len(r.stderr.encode()) <= 1024 ** 2,
            f"{tier}: stdout {len(r.stdout.encode())}B / stderr {len(r.stderr.encode())}B <= 1 MiB 각각")
        listing, body = _read_out_volume(vol)
        names = [l.split()[-1] for l in listing.splitlines()[1:]
                 if l.strip() and l.split()[-1] not in (".", "..")]
        rec("C", f"C4{tier[0]}", names == ["submission.json"],
            f"{tier}: 출력 볼륨 루트에 submission.json 하나만 ({names})",
            "RUNTIME.md: submission.json 외의 파일·디렉터리·링크 금지")
        mode_ok = any(l.startswith("-rw-r--r--") and l.rstrip().endswith("submission.json")
                      for l in listing.splitlines())
        rec("C", f"C5{tier[0]}", mode_ok, f"{tier}: 출력 파일 권한 0644")
        try:
            payload = json.loads(body)
            sub = parse_submission(payload)
            ok = (sub.tier == tier and sub.policy_id == POLICY_ID
                  and sub.challenge_id == CHALLENGE_ID and sub.split == "public-train-dev")
            ids = [d.episode_id for d in sub.decisions]
            complete = sorted(ids) == sorted(e["episode_id"] for e in eps)
            models_ok = all(d.model_id in MODELS for d in sub.decisions)
            rec("C", f"C6{tier[0]}", ok, f"{tier}: v1 스키마·tier·policy_id·challenge_id·split 일치")
            rec("C", f"C7{tier[0]}", complete and len(ids) == len(set(ids)),
                f"{tier}: 문항 {len(ids)}개, 누락·중복·추가 없음")
            rec("C", f"C8{tier[0]}", models_ok, f"{tier}: model_id 전부 허용 3종")
            outs[tier] = {d.episode_id: d.model_id for d in sub.decisions}
        except Exception as exc:
            rec("C", f"C6{tier[0]}", False, f"{tier}: 출력 파싱 실패 {exc}")
        sh(["docker", "volume", "rm", "-f", vol])

    # error path: malformed input must exit 2, not 0 and not crash silently
    bad = work / "bad"
    (bad).mkdir(parents=True, exist_ok=True)
    (bad / "inputs.json").write_text("{ not json", encoding="utf-8")
    r, _ = _run_container(image, bad, "ossp-compl-bad", "fast")
    rec("C", "C9", r.returncode == 2,
        f"형식 오류 입력에 종료코드 2 (실제 {r.returncode})",
        "RUNTIME.md: 입력·인자·형식 오류는 종료 코드 2")
    sh(["docker", "volume", "rm", "-f", "ossp-compl-bad"])

    # messages[] encoding, which the public data never exercises
    msg_dir = work / "msg"
    _write_input(msg_dir / "inputs.json", [
        {"episode_id": f"m{i}", "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Question: what is {i}+{i}?\nA. {2*i}\nB. {2*i+1}"}]}
        for i in range(40)])
    r, _ = _run_container(image, msg_dir, "ossp-compl-msg", "balanced")
    _, body = _read_out_volume("ossp-compl-msg")
    try:
        sub = parse_submission(json.loads(body))
        rec("C", "C10", len(sub.decisions) == 40 and all(d.model_id in MODELS for d in sub.decisions),
            f"messages[] 형식 입력 처리 ({len(sub.decisions)}개 결정)")
    except Exception as exc:
        rec("C", "C10", False, f"messages[] 형식 실패: {exc}")
    sh(["docker", "volume", "rm", "-f", "ossp-compl-msg"])
    shutil.rmtree(work, ignore_errors=True)
    return outs


# ---------------------------------------------------------------- D audit
def group_d(image: str | None, have_docker: bool, base: dict) -> None:
    print("\nD. 규칙 준수 감사 시뮬레이션  (CHALLENGE_RULES.md / ENFORCEMENT.md)")
    rt = (ROOT / "src" / "ossp_router" / "lpb.py").read_text(encoding="utf-8")
    # look for actual reads, not the word in prose: the module docstring says it
    # does NOT see outcomes, and a substring match flags that as a violation.
    # Patterns must name the outcome API specifically. A bare ["models"] match
    # is wrong: the artifact carries its own model-id list as blob["models"],
    # which is not an outcome read.
    reads = [p for p in ("load_outcomes", "parse_outcomes", "outcomes.json",
                         "OutcomeBatch", "ModelOutcome", ".score", "num_generations",
                         "output_tokens", "input_tokens")
             if p in rt]
    rec("D", "D1", not reads, f"런타임 모듈이 outcome API 를 호출하지 않음 {reads}",
        "CHALLENGE_RULES.md: 문항별 평가 결과는 라우터에 전달되지 않음")
    for bad in ("episode_id", "split", "challenge_id"):
        # allowed only for writing the decision back, never as a routing feature
        uses = re.findall(rf"\b{bad}\b", rt)
        rec("D", f"D2-{bad[:4]}", True,
            f"{bad} 등장 {len(uses)}회 — 선택 로직 사용 여부는 D3~D5 동적 검사로 판정")
    if not (have_docker and image and base):
        rec("D", "D3", None, "컨테이너 결과 없음 — ID·순서 감사 생략")
        return
    from ossp_router.protocol import load_input, parse_submission

    inputs = load_input(ROOT / "data" / "materialized" / "dev" / "inputs.json")
    eps = [{"episode_id": e.episode_id,
            **({"prompt": e.prompt} if e.prompt is not None
               else {"messages": [{"role": m.role, "content": m.content} for m in e.messages]})}
           for e in inputs.episodes]
    work = Path(tempfile.mkdtemp(prefix="ossp-audit-"))

    def run_variant(name: str, episodes: list, cid=CHALLENGE_ID, split="probe"):
        d = work / name
        _write_input(d / "inputs.json", episodes, challenge_id=cid, split=split)
        vol = f"ossp-audit-{name}"
        r, _ = _run_container(image, d, vol, "balanced")
        _, body = _read_out_volume(vol)
        sh(["docker", "volume", "rm", "-f", vol])
        sub = parse_submission(json.loads(body))
        return {x.episode_id: x.model_id for x in sub.decisions}

    ref = run_variant("ref", eps)
    rec("D", "D3", ref == run_variant("rep", eps), "같은 입력 반복 실행 결정 동일 (결정성)")

    import random
    rng = random.Random(20260827)
    toks = [f"audit-{rng.getrandbits(48):012x}" for _ in eps]
    renamed = [{**e, "episode_id": t} for e, t in zip(eps, toks)]
    got = run_variant("id", renamed)
    mism = [e["episode_id"] for e, t in zip(eps, toks) if ref[e["episode_id"]] != got[t]]
    rec("D", "D4", not mism, f"episode_id 전면 치환 후 선택 동일 (불일치 {len(mism)})",
        "ENFORCEMENT.md: 콘텐츠가 같은데 ID·순서만 바꿨을 때 선택이 달라지면 공정성 검토")

    shuf = list(eps)
    rng.shuffle(shuf)
    got = run_variant("order", shuf)
    mism = [k for k in ref if ref[k] != got[k]]
    rec("D", "D5", not mism, f"입력 순서 셔플 후 선택 동일 (불일치 {len(mism)})")

    got = run_variant("meta", eps, cid=CHALLENGE_ID, split="totally-different-split")
    mism = [k for k in ref if ref[k] != got[k]]
    rec("D", "D6", not mism, f"split 값 변경 후 선택 동일 (불일치 {len(mism)})",
        "CHALLENGE_RULES.md: split 에 따라 선택을 바꾸는 방식 금지")
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--skip-runtime", action="store_true")
    ap.add_argument("--verify-rebuild", action="store_true",
                    help="현재 커밋에서 이미지를 재빌드해 제출 다이제스트와 비교 (수 분)")
    args = ap.parse_args()
    global VERIFY_REBUILD
    VERIFY_REBUILD = args.verify_rebuild

    print("=" * 78)
    print("SKT OSSP 2026 제출 규격 준수 하네스")
    print("=" * 78)
    meta = group_a()
    have_docker = group_b(args.image, meta)
    base = {} if args.skip_runtime else group_c(args.image, have_docker)
    if not args.skip_runtime:
        group_d(args.image, have_docker, base)
    group_e()
    group_f()

    print("\n" + "=" * 78)
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    skips = [r for r in RESULTS if r[2] == "SKIP"]
    dq = [r for r in fails if r[0] in ("A", "D", "E")]
    print(f"총 {len(RESULTS)}개 검사 · PASS {len(RESULTS)-len(fails)-len(skips)} · "
          f"FAIL {len(fails)} · SKIP {len(skips)}")
    if dq:
        print("\n★ 실격 위험 항목:")
        for g, cid, _, d in dq:
            print(f"    {g}/{cid}  {d}")
    if fails and not dq:
        print("\n등급 손실 위험 항목:")
        for g, cid, _, d in fails:
            print(f"    {g}/{cid}  {d}")
    if skips:
        print("\n미검증(SKIP) 항목 — 반드시 별도 확인:")
        for g, cid, _, d in skips:
            print(f"    {g}/{cid}  {d}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

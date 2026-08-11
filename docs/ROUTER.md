<!--
SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
SPDX-License-Identifier: Apache-2.0
-->

# 참가자 라우터 (LPB Router)

`router-run` 진입점은 `src/ossp_router/lpb.py` 입니다.
런타임은 **Python 표준 라이브러리만** 사용하며, 학습에만 NumPy를 씁니다.

## 설계 요약

이 과제에서 등급 점수를 0으로 만드는 것은 라우팅 실패가 아니라 **비용 초과**입니다.
공개 Dev에서 `axk1-think` 의 문항당 비용은 all-light 기준선의 **22.6배**이고,
Fast 등급의 여유분은 전체의 0.25배뿐이라 승급 가능한 문항은 약 1%입니다.
주최측 `hash-regex` baseline 도 공개 Dev 3.985 → 채점셋 약 4.2 로 Premium 0점을
받았습니다(`baselines/README.md`). 그래서 순서를 비용 → 품질로 둡니다.

1. **크레딧이 아니라 토큰을 예측한다.** 입력·출력 토큰 수를 모델별로 로그공간에서
   회귀한 뒤, 공개된 단가로 비용을 조립합니다. 단가는 정확히 알려져 있고 불확실한
   것은 토큰 수뿐이기 때문입니다.
2. **Duan smearing 보정.** `axk1-think` 출력 토큰은 평균 3,185 / 중앙값 1,499 로
   꼬리가 두껍습니다. 로그 적합의 `exp()` 는 기하평균을 주는데 예산은 산술평균으로
   과금되므로, smearing 계수로 되돌립니다.
3. **배치 전체로 배분한다.** `docs/RUNTIME.md` 는 입력 전체를 한 번에 전달하고
   예산은 그 배치의 총합에 걸립니다. 문항마다 (비용, 점수) 의 상위 오목 포락선을
   만들고, **크레딧당 품질 이득**이 큰 승급부터 한도까지 사들입니다.
4. **여유는 초과분에만 적용한다.** all-light 는 정의상 1.0배라 초과할 수 없으므로
   위험한 것은 초과분뿐입니다. 총 한도에 계수를 곱하면 Fast(1.25배)는 가용 여유의
   대부분을 잃고 Premium(4.0배)은 조금만 잃어 — 가중치가 가장 큰 등급이 가장 크게
   손해 봅니다.
5. **외삽을 막는다.** 로그선형 적합은 학습 범위 밖에서 무한정 외삽합니다. 보정 전
   7만 자 프롬프트에서 `ax31-light` 의 출력 토큰이 **155만 개**(생성 조건 32,768의 47배)로
   예측됐고, 같은 프롬프트에서 `ax31` 은 **light 의 0.04배 비용**으로 나왔습니다.
   그러면 배분기가 "공짜 승급"을 무한히 사들입니다. 두 가지로 막습니다.
   - 예측 토큰 수를 **학습에서 관측된 범위로 클램프**
   - 공개 단가 사다리(`ax31-light` < `ax31` < `axk1-think`)를 따라
     **비싼 모델이 싼 모델보다 싸게 예측되지 않도록 하한**을 겁니다.
     공개 Dev 실측에서 `ax31` 이 `ax31-light` 보다 싼 경우는 0.35%, `axk1-think` 는 0건입니다.
6. **점수 예측은 2PL 문항반응모형.** `p_m = sigmoid(a_m · (θ(x) − b_m))`,
   프롬프트 능력 `θ(x)` 는 공유하고 모델마다 파라미터 2개만 둡니다.
   독립 회귀 3개와 grouped 5-fold CV 를 fold 시드 3개로 비교했을 때 3/3 에서
   앞섰고, 시드 간 변동폭이 **0.0011 대 0.0230** 이었습니다. 비공개 평가셋의
   구성이 공개되지 않는 과제에서는 평균보다 이 안정성이 중요합니다.

## 규칙 준수

- 선택에 사용하는 정보는 **프롬프트 내용과 등급뿐**입니다.
- `episode_id`, `split`, `challenge_id`, 입력 순서를 읽지 않습니다.
  특성 해싱과 동률 처리에 쓰는 키는 모두 프롬프트 내용에서 계산합니다.
- 파이썬 `hash()` 는 프로세스마다 salt 가 달라 재실행 시 다른 결과를 낼 수 있으므로
  사용하지 않고 `zlib.crc32` 를 씁니다.
- 등급별 예산 배분은 배치 전체를 대상으로 하지만, ID·순서에는 불변입니다.
  `tests/test_lpb_router.py` 가 ID 치환·순서 셔플·반복 실행 결정성을 검사합니다.
- 네트워크를 사용하지 않고 실행 중 아무것도 내려받지 않습니다.

## 포함한 학습 파일

`docs/SUBMISSION.md` 의 기록 요건에 따릅니다.

| 항목 | 값 |
| --- | --- |
| 이름 | `src/ossp_router/resources/lpb-artifact.v1.json` |
| 용도 | 모델별 점수·토큰 수 예측 계수와 등급별 예산 여유 계수 |
| 생성 도구 | `tools/train_lpb.py` (이 저장소, Apache-2.0) |
| 업스트림 URL | 없음 — 외부에서 받은 모델·가중치가 아니라 이 저장소가 공개 Train 에서 직접 적합한 계수입니다 |
| 학습 입력 | `data/materialized/train/inputs.json` (SHA-256 `029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4`) |
| 학습 결과 | `data/train/outcomes.json` (SHA-256 `0a35c1ce83e074ffc8e470d5c4f49d35765371384ecff3db91bad9de4ef2ffe7`) |
| 산출물 SHA-256 | `120e1dd951cbfcf1bd18b22d356430aaa2c2b11a0508ea06043d56f90e15f1ac` |
| 크기 | 76,750 바이트 |
| 라이선스 | Apache-2.0 (이 저장소와 동일) |
| 학습 의존성 | NumPy (BSD-3-Clause) — **학습 전용이며 제출 이미지에 포함되지 않습니다** |

공개 Dev 는 계수 학습에 사용하지 않았습니다. 등급별 여유 계수는 공개 Dev 점수를
최대화하는 값이 아니라, Train 내부 **out-of-fold 실현 비용비**가 한도의 90% 이하로
유지되는 값으로 정했습니다.

## 재현

```console
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py

python3 -m pip install -r baselines/requirements-train.txt
PYTHONPATH=src python3 tools/train_lpb.py \
  --train-input data/materialized/train/inputs.json \
  --train-outcomes data/train/outcomes.json \
  --artifact src/ossp_router/resources/lpb-artifact.v1.json \
  --report build/lpb-train-report.json

for tier in fast balanced premium; do
  PYTHONPATH=src python3 -m ossp_router.lpb \
    --input data/materialized/dev/inputs.json \
    --tier "$tier" --output "build/lpb/dev/$tier.json"
done

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --submissions build/lpb/dev \
  --report build/lpb-dev-report.json
```

## 현재 성적

공개 Dev 880문항, 학습은 공개 Train 1,760문항만 사용 (Dev 는 held-out).

| 등급 | 점수 | 비용 비율 / 한도 | 한도 사용률 | 예산 통과 |
| --- | ---: | ---: | ---: | --- |
| Fast | 0.648580 | 1.093871 / 1.25 | 87.5% | 통과 |
| Balanced | 0.689773 | 1.723943 / 2.0 | 86.2% | 통과 |
| Premium | 0.719602 | 3.172569 / 4.0 | 79.3% | 통과 |
| **최종** | **0.682244** | | | |

참고: all-light 0.619318 · `prompt-heuristic` 0.655341 · `feature-budget` 0.643011 ·
`hash-regex` 0.695369 (`baselines/README.md`). 전지적 oracle 상한은 약 0.807 입니다.

`hash-regex` 는 최종 점수가 0.0135 높지만 세 등급의 한도 사용률이 각각
**98.9% / 98.1% / 99.6%** 이고 채점셋에서 Premium 이 실제로 초과했습니다.
이 라우터는 세 등급 모두 `near_budget` 경보(95%) 아래에 있습니다.

## 자원 한도 검사

`tools/check_runtime.py`, 공개 Train+Dev 전체 2,640문항, `linux/arm64` 이미지,
CPU 2 · 메모리 2 GiB · 프로세스 32 · 네트워크 없음 · 읽기 전용 루트.

| 등급 | 실행 시간 | 한도 | 결과 |
| --- | ---: | ---: | --- |
| Fast | 37.3초 | 90초 | PASS |
| Balanced | 35.1초 | 90초 | PASS |
| Premium | 39.9초 | 90초 | PASS |

보고서: `"passed": true`. 측정은 x86_64 호스트에서 arm64 를 에뮬레이션한 값이므로
공식 Apple Silicon 장비의 네이티브 실행보다 **느린 쪽**입니다.

> **범위 고지.** 위 arm64 측정은 외삽 방지 보정(위 §5)을 넣기 **직전** 빌드의 값입니다.
> 보정 이후 같은 하네스를 다시 돌리려 했으나 호스트의 Docker 가 arm64 binfmt 등록을
> 해제해 `exec format error` 로 실행되지 않았고, 복구에는 호스트 설정을 바꾸는
> privileged 컨테이너가 필요해 수행하지 않았습니다. 보정이 추가한 연산은 문항당
> 클램프와 비교 몇 번뿐이며, 같은 2,640문항에 대한 네이티브(x86) 실행 시간은
> 보정 전후 모두 **2.2~2.4초**로 변화가 없습니다. 그래도 이 표는 재측정 전까지
> **직전 빌드 기준**으로 읽어야 합니다.

네이티브(x86, 컨테이너 없음) 참고값: 2,640문항 등급당 2.2~2.4초.

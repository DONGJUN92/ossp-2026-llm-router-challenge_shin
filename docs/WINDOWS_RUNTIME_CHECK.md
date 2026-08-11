<!--
SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
SPDX-License-Identifier: Apache-2.0
-->

# Windows 에서 공식 런타임 검사기 실행하기

[`README.md`](../README.md) 는 제출 전 확인 절차로 `tools/check_runtime.py` 를
안내하지만, 이 도구는 Windows 의 Python 으로는 실행되지 않습니다. 이 문서는 그
이유와, 코드를 고치지 않고 원본 그대로 실행하는 방법을 설명합니다.

결과부터: 이 저장소의 제출 이미지는 이 방법으로 **세 등급 모두 통과**했습니다.

```
report_type : participant-public-runtime-check
passed      : True
fast     PASS  31.264초 / 90초
balanced PASS  30.825초 / 90초
premium  PASS  31.206초 / 90초
```

## 무엇이 막는가

`tools/check_runtime.py` 는 `ossp_router.orchestrator` 를 거쳐
`ossp_router.runtime` 를 import 하고, 그 모듈이 POSIX 전용 기능 다섯 가지를
사용합니다.

| 의존 | 사용처 | 하는 일 |
| --- | ---: | --- |
| `fcntl.flock` | 2곳 | 같은 작업 공간에서 동시 실행 방지 |
| `resource` | 18곳 | 자원 한도 설정과 사용량 측정 |
| `os.O_NOFOLLOW` | 9곳 | 출력 파일을 열 때 심볼릭 링크 우회 차단 |
| `os.geteuid` | 3곳 | 실행 권한 확인 |
| `signal.SIGKILL` | 1곳 | 시간 초과 컨테이너 강제 종료 |

Windows 에는 이 다섯 가지가 없습니다.

### 가짜 모듈로 대체하지 않는 이유

빈 `fcntl` 모듈을 만들어 import 만 통과시키는 방법이 흔히 쓰입니다. 이 저장소에서는
택하지 않았습니다. 다섯 중 셋(`O_NOFOLLOW`, `geteuid`, `flock`)이 **검사 그 자체**
이기 때문입니다. 무동작으로 바꾸면 검사기는 통과하지만, 그 통과가 심볼릭 링크
공격 차단이나 권한 확인에 대해 아무것도 보증하지 않습니다. 확인되지 않은 것을
확인된 것처럼 보이게 만드는 절차는, 검사를 하지 않는 것보다 나쁩니다.

## 방법: 진짜 Linux 에서 원본 그대로 실행

Docker Desktop 의 데몬을 공유하는 Linux 컨테이너 안에서 검사기를 그대로 돌립니다.
`fcntl`, `resource`, `O_NOFOLLOW`, `geteuid`, `SIGKILL` 이 모두 진짜로 동작합니다.

한 가지 함정이 있습니다. **컨테이너 안에서 요청한 bind mount 는 컨테이너가 아니라
데몬이 해석합니다.** 검사기가 `--mount type=bind,src=/어떤/경로` 를 만들면 그 경로는
데몬이 있는 VM 안에서 찾습니다. 그래서 저장소를 **데몬이 보는 것과 같은 문자열의
경로**에 마운트해야 합니다. Docker Desktop 은 호스트 드라이브를 VM 안에서
`/run/desktop/mnt/host/<드라이브>/...` 로 노출합니다.

`HOME` 도 함께 옮겨야 합니다. 검사기는 임시 입력 디렉터리를 `/tmp` 가 아니라
`Path.home()` 아래에 만들기 때문에, `HOME` 이 컨테이너 내부 경로면 데몬이 그
디렉터리를 볼 수 없습니다.

[`tools/check_runtime_windows.ps1`](../tools/check_runtime_windows.ps1) 이 이
계산을 대신합니다.

```console
powershell -File tools\check_runtime_windows.ps1 -Image ghcr.io/dongjun92/ossp-router:submission
```

등급별로 여러 번 실행해 결정성까지 확인하려면 `-Repetitions` 를 씁니다. 반복
사이에 출력 SHA-256 이 같아야 합니다.

```console
powershell -File tools\check_runtime_windows.ps1 -Image ossp-router:local -Repetitions 5
```

미리 갖춰야 할 것은 세 가지입니다.

1. Docker Desktop 실행 중
2. `data/materialized/{train,dev}/inputs.json` — [`README.md`](../README.md) 의
   공개 자료 materialization 을 먼저 수행
3. arm64 에뮬레이션 등록 (아래)

## arm64 에뮬레이션은 조용히 사라진다

제출 이미지는 `linux/arm64` 이고 개발 장비는 대개 amd64 이므로, 실행하려면 QEMU
binfmt 핸들러가 등록되어 있어야 합니다. 이 등록은 Docker Desktop 재시작 등으로
예고 없이 사라질 수 있습니다. 사라진 상태에서 검사하면 **정상적인 제출물이**

```
exec /usr/local/bin/python3: exec format error
```

로 실패합니다. 제출물 문제로 오해하기 쉽습니다.

확인은 반드시 이 명령으로 하십시오.

```console
docker run --rm --platform linux/arm64 alpine:3.23 uname -m
```

`aarch64` 가 나와야 정상입니다. 복구는 다음과 같습니다.

```console
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

> **`docker buildx inspect` 로 판단하지 마십시오.** 에뮬레이션이 사라진 상태에서도
> `linux/arm64` 를 계속 나열합니다. buildx 가 보고하는 플랫폼은 빌더 컨테이너 안의
> QEMU 이고, 컨테이너 실행에 쓰이는 데몬의 binfmt 등록과는 별개이기 때문입니다.

스크립트에 `-InstallBinfmt` 를 주면 없을 때 자동으로 설치합니다. 권한 있는
컨테이너를 한 번 실행하므로 기본값은 꺼져 있습니다.

## 이 방법으로 확인되는 것과 확인되지 않는 것

확인되는 것은 [`RUNTIME.md`](RUNTIME.md) 가 정의한 경계 전부입니다. CPU 2코어,
메모리 2 GiB, swap 합계 2 GiB, 프로세스·스레드 32개, 출력 4 MiB, 임시 공간
256 MiB, 등급당 90초, 네트워크 차단, 읽기 전용 루트, 비특권 사용자, 그리고 반복
실행 사이의 출력 동일성입니다.

확인되지 않는 것은 **실행 시간의 절대값**입니다.
[`APPLE_SILICON_MEASUREMENT.md`](APPLE_SILICON_MEASUREMENT.md) 가 정한 공식 측정은
Apple Silicon 의 네이티브 `linux/arm64` 에서 이루어지지만, 이 방법은 amd64 위에서
QEMU 로 변환해 실행하므로 훨씬 느립니다. 따라서 여기서 얻은 시간은 한도 대비 여유를
보수적으로 가늠하는 용도로만 쓰고, [`SCORING.md`](SCORING.md) 의 동점 레이턴시와
비교해서는 안 됩니다.

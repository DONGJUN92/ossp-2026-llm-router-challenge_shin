# SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Windows 에서 tools/check_runtime.py (SKT 참가자용 런타임 검사기) 를 실행합니다.

.DESCRIPTION
check_runtime.py 는 fcntl, resource, os.O_NOFOLLOW, os.geteuid, signal.SIGKILL 을
쓰므로 Windows 의 Python 으로는 import 조차 되지 않습니다. 그 다섯 가지를 가짜
모듈로 대체하는 방법은 택하지 않았습니다. 셋은 보안 검사 자체(심볼릭 링크 차단,
권한 확인, 동시 실행 방지)라서, 무동작으로 바꾸면 통과는 하되 그 통과가 아무것도
보증하지 않기 때문입니다.

대신 Docker Desktop 데몬을 공유하는 Linux 컨테이너 안에서 원본 그대로 실행합니다.
컨테이너 안에서 요청한 bind mount 는 컨테이너가 아니라 데몬이 해석하므로, 데몬이
보는 경로와 같은 문자열로 저장소를 마운트해야 합니다. Docker Desktop 은 호스트
드라이브를 VM 안에서 /run/desktop/mnt/host/<드라이브>/... 로 노출하며, 이 스크립트가
그 경로를 계산해 맞춰 줍니다.

HOME 도 함께 옮깁니다. 검사기는 임시 입력 디렉터리를 /tmp 가 아니라 Path.home()
아래에 만들기 때문에(check_runtime.py 의 tempfile.TemporaryDirectory(dir=...)),
HOME 이 컨테이너 내부 경로면 데몬이 그 디렉터리를 볼 수 없습니다.

.PARAMETER Image
검사할 로컬 이미지. 태그 또는 다이제스트.

.PARAMETER Repetitions
등급별 반복 횟수 1~5. 기본 1. 반복은 결정성 확인에 쓰입니다.

.PARAMETER Report
JSON 보고서를 저장할 경로. 생략하면 build/runtime-check-report.json.

.PARAMETER InstallBinfmt
arm64 에뮬레이션이 등록되어 있지 않으면 자동으로 설치합니다. 권한 있는(privileged)
컨테이너를 한 번 실행하므로 기본값은 꺼져 있습니다.

.EXAMPLE
powershell -File tools\check_runtime_windows.ps1 -Image ghcr.io/dongjun92/ossp-router:submission

.EXAMPLE
powershell -File tools\check_runtime_windows.ps1 -Image ossp-router:local -Repetitions 5
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [ValidateRange(1, 5)][int]$Repetitions = 1,
    [string]$Report,
    [switch]$InstallBinfmt,
    [string]$CliImage = "docker:cli"
)

$ErrorActionPreference = "Stop"

function Fail($message) {
    Write-Host "오류: $message" -ForegroundColor Red
    exit 2
}

# ---------------------------------------------------------------- 저장소 위치
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.TrimEnd('\')
if (-not (Test-Path (Join-Path $root "tools\check_runtime.py"))) {
    Fail "tools\check_runtime.py 를 찾지 못했습니다. 저장소 안에서 실행하십시오."
}
foreach ($needed in @("data\materialized\train\inputs.json", "data\materialized\dev\inputs.json")) {
    if (-not (Test-Path (Join-Path $root $needed))) {
        Fail "$needed 이 없습니다. README 의 공개 자료 materialization 을 먼저 수행하십시오."
    }
}

if ($root -notmatch '^[A-Za-z]:\\') {
    Fail "저장소가 로컬 드라이브에 있어야 합니다 (현재: $root). UNC 경로는 VM 에 노출되지 않습니다."
}
$drive = $root.Substring(0, 1).ToLower()
$vmRoot = "/run/desktop/mnt/host/$drive/" + ($root.Substring(3) -replace '\\', '/')

# ---------------------------------------------------------------- 사전 점검
try { docker version --format '{{.Server.Os}}' | Out-Null }
catch { Fail "Docker 데몬에 접속할 수 없습니다. Docker Desktop 을 실행하십시오." }

Write-Host "저장소      : $root"
Write-Host "VM 경로     : $vmRoot"
Write-Host "이미지      : $Image"

$arch = (docker run --rm --platform linux/arm64 alpine:3.23 uname -m 2>&1 | Select-Object -Last 1).ToString().Trim()
if ($arch -ne "aarch64") {
    Write-Host "arm64 에뮬레이션: 없음 ($arch)" -ForegroundColor Yellow
    if ($InstallBinfmt) {
        Write-Host "binfmt 핸들러를 설치합니다..."
        docker run --privileged --rm tonistiigi/binfmt --install arm64 | Out-Null
        $arch = (docker run --rm --platform linux/arm64 alpine:3.23 uname -m 2>&1 | Select-Object -Last 1).ToString().Trim()
        if ($arch -ne "aarch64") { Fail "binfmt 설치 후에도 arm64 실행이 되지 않습니다." }
    }
    else {
        Fail @"
arm64 에뮬레이션이 등록되어 있지 않습니다. 이 상태로 검사하면 정상적인 제출물도
'exec format error' 로 실패합니다. 다음 중 하나를 하십시오.

  1) 이 스크립트를 -InstallBinfmt 와 함께 다시 실행
  2) 직접 설치:  docker run --privileged --rm tonistiigi/binfmt --install arm64

주의: docker buildx inspect 는 이 상황에서도 linux/arm64 를 계속 나열합니다.
빌더 컨테이너 안의 QEMU 는 별개이기 때문이며, 판정 근거로 쓰면 안 됩니다.
확인은 항상 docker run --rm --platform linux/arm64 alpine:3.23 uname -m 으로 하십시오.
"@
    }
}
Write-Host "arm64 에뮬레이션: 정상 ($arch)"

# ---------------------------------------------------------------- 작업 공간
$work = Join-Path $root "build\windows-runtime-check"
$checkHome = Join-Path $work "home"
New-Item -ItemType Directory -Force $work, $checkHome | Out-Null
if (-not $Report) { $Report = Join-Path $root "build\runtime-check-report.json" }
New-Item -ItemType Directory -Force (Split-Path $Report) | Out-Null

$reportFull = (Resolve-Path (Split-Path $Report)).Path.TrimEnd('\') + '\' + (Split-Path $Report -Leaf)
if ($reportFull.Substring(0, 1).ToLower() -ne $drive) {
    Fail "보고서 경로는 저장소와 같은 드라이브여야 합니다 ($reportFull)."
}
$vmReport = "/run/desktop/mnt/host/$drive/" + ($reportFull.Substring(3) -replace '\\', '/')
$vmHome = $vmRoot + "/build/windows-runtime-check/home"

# 컨테이너 안에서 실행할 스크립트. PowerShell 이 $(...) 를 먼저 해석하지 않도록
# 파일로 넘긴다.
$inner = @"
#!/bin/sh
set -u
apk add --no-cache python3 >/dev/null 2>&1 || exit 90
cd '$vmRoot' || exit 91
# 검사기는 --platform 을 공식 장비(arm64 네이티브) 기준으로만 전달한다. amd64
# 데몬에서는 이 변수가 있어야 CLI 가 arm64 를 명시해 QEMU 변환이 걸린다.
export DOCKER_DEFAULT_PLATFORM=linux/arm64
PYTHONPATH=src python3 tools/check_runtime.py \
  --image '$Image' \
  --repetitions $Repetitions \
  --report '$vmReport'
"@
$innerPath = Join-Path $work "run_check.sh"
[System.IO.File]::WriteAllText($innerPath, ($inner -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))

Write-Host "반복        : $Repetitions 회/등급"
Write-Host "보고서      : $reportFull"
Write-Host ("-" * 70)

docker run --rm `
    -v "/var/run/docker.sock:/var/run/docker.sock" `
    -v "${root}:${vmRoot}" `
    -e "HOME=$vmHome" `
    --entrypoint sh $CliImage "$vmRoot/build/windows-runtime-check/run_check.sh"
$code = $LASTEXITCODE

Write-Host ("-" * 70)
if ($code -eq 0) {
    Write-Host "결과: 통과 (exit 0). 보고서: $reportFull" -ForegroundColor Green
}
else {
    Write-Host "결과: 실패 (exit $code)" -ForegroundColor Red
}
Write-Host @"

실행 시간은 참고값입니다. 공식 평가는 Apple Silicon 의 네이티브 linux/arm64 에서
수행하지만 여기서는 amd64 위에서 QEMU 로 변환해 실행하므로 실제보다 느립니다.
경계·형식·결정성 검증은 그대로 유효합니다.
"@
exit $code

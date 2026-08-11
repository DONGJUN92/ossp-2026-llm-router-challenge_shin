<!--
SPDX-FileCopyrightText: Copyright 2026 DONGJUN92
SPDX-License-Identifier: Apache-2.0
-->

# 이미지 올리기 — 처음부터 끝까지

`docs/SUBMISSION.md` 는 "이미지를 공개 레지스트리에 올리고 그 다이제스트를 제출"하라고
요구합니다. 이 문서는 그 과정을 **터미널을 처음 써 보는 사람 기준**으로 나눈 것입니다.

- 계정: `DONGJUN92`
- 이미지 주소: `ghcr.io/dongjun92/ossp-router` (**전부 소문자**. 계정이 대문자여도 소문자로 씁니다)
- 저장소 폴더: `C:\Users\holiy\Downloads\test_Opensource\work\fork`

전체 소요 시간은 처음이면 20~30분, 두 번째부터는 5분 정도입니다.

---

## 0단계 · Docker Desktop 켜기

1. 시작 버튼 → `Docker` 입력 → **Docker Desktop** 실행
2. 창 왼쪽 아래 고래 아이콘 옆 상태가 **초록색 `Engine running`** 이 될 때까지 기다립니다
   (처음 켜면 1~2분 걸립니다)

이게 안 되어 있으면 아래 모든 `docker` 명령이 실패합니다.

---

## 1단계 · 명령 입력창(PowerShell) 열기

1. 키보드에서 **`Windows 키`** 를 누릅니다
2. `powershell` 이라고 입력합니다
3. **Windows PowerShell** 을 클릭합니다

파란색(또는 검은색) 창이 열리고 이렇게 한 줄이 보이면 준비된 상태입니다.

```
PS C:\Users\holiy>
```

**명령을 붙여넣는 방법**: 이 문서에서 명령을 마우스로 긁어 복사(`Ctrl+C`)한 뒤,
PowerShell 창에서 **마우스 오른쪽 버튼을 한 번 클릭**하면 붙여집니다.
그다음 **`Enter`** 를 눌러야 실행됩니다. (`Ctrl+V` 도 되는 경우가 많습니다.)

---

## 2단계 · 저장소 폴더로 이동

아래 한 줄을 붙여넣고 `Enter`:

```powershell
cd C:\Users\holiy\Downloads\test_Opensource\work\fork
```

**정상이면** 아무 메시지 없이 맨 앞 표시가 이렇게 바뀝니다:

```
PS C:\Users\holiy\Downloads\test_Opensource\work\fork>
```

이 창을 끝까지 그대로 씁니다. 창을 닫으면 2단계부터 다시 하세요.

---

## 3단계 · 로그인이 살아 있는지 확인

이미 `DONGJUN92` 로 로그인한 기록이 있습니다. 토큰이 아직 유효한지만 봅니다.

```powershell
docker pull ghcr.io/dongjun92/ossp-router:submission
```

나오는 메시지로 판단합니다.

| 나온 메시지 | 뜻 | 다음 |
| --- | --- | --- |
| `manifest unknown` / `not found` | **로그인 정상.** 아직 올린 이미지가 없을 뿐 | 5단계로 |
| `unauthorized` / `denied` | 토큰이 만료됐거나 권한 부족 | 4단계로 |

---

## 4단계 · 토큰 만들기 (3단계에서 `unauthorized` 가 나온 경우만)

GitHub 는 레지스트리 로그인에 **비밀번호를 받지 않고 토큰을 받습니다.**

1. 브라우저에서 <https://github.com/settings/tokens> 를 엽니다
2. 오른쪽 위 **`Generate new token`** → **`Generate new token (classic)`** 선택
   (Fine-grained 말고 **classic** 입니다)
3. **Note** 칸에 `ghcr-ossp` 라고 적습니다 (아무 이름이나 됩니다)
4. **Expiration** 은 `90 days` 정도로 둡니다 (대회 마감 2026-08-27 이후까지)
5. 아래 체크박스 목록에서 **`write:packages`** 하나만 체크합니다
   (체크하면 `read:packages` 는 자동으로 같이 켜집니다)
6. 맨 아래 초록색 **`Generate token`** 버튼 클릭
7. 화면에 `ghp_` 로 시작하는 긴 문자열이 나옵니다. **오른쪽 복사 아이콘을 눌러 복사**하세요

> 이 화면을 벗어나면 그 문자열은 다시 볼 수 없습니다. 못 봤으면 6번부터 다시 하면 됩니다.

이제 PowerShell 창으로 돌아와서:

```powershell
docker login ghcr.io -u DONGJUN92
```

`Password:` 라고 물어보면 **GitHub 비밀번호가 아니라 방금 복사한 토큰**을 붙여넣고
`Enter` 를 누릅니다.

> 붙여넣어도 화면에 아무것도 안 보이는 게 정상입니다. 비밀번호 입력은 원래 표시되지
> 않습니다. 그냥 `Enter` 를 누르세요.

**정상이면** `Login Succeeded` 가 나옵니다.

이 토큰은 저에게 보여주지 마세요. 이 명령은 직접 실행하셔야 합니다.

---

## 5단계 · 이미지 만들기 (빌드)

```powershell
docker build --pull --platform linux/arm64 --provenance=false --sbom=false --file container/Dockerfile --tag ghcr.io/dongjun92/ossp-router:submission .
```

- 맨 뒤의 **점(`.`)도 명령의 일부**입니다. 빠뜨리지 마세요.
- 1~3분 걸립니다. 글자가 계속 올라가는 게 정상입니다.

**정상이면** 마지막에 이런 줄이 나옵니다:

```
naming to ghcr.io/dongjun92/ossp-router:submission  done
```

확인:

```powershell
docker image inspect ghcr.io/dongjun92/ossp-router:submission --format '{{.Os}}/{{.Architecture}}'
```

**`linux/arm64`** 가 나와야 합니다. `amd64` 가 나오면 `--platform` 을 빠뜨린 것이니
5단계를 다시 하세요.

---

## 6단계 · 올리기 (푸시)

```powershell
docker push ghcr.io/dongjun92/ossp-router:submission
```

- 1~3분 걸립니다.
- **정상이면** 마지막 줄에 `digest: sha256:...` 과 크기가 나옵니다.
- `denied` 가 나오면 4단계 토큰에 `write:packages` 가 빠진 것입니다.

---

## 7단계 · 공개로 바꾸기 ★ 빠뜨리면 심사 불가

**처음 올린 패키지는 비공개가 기본값입니다.** 대회 규칙은 "별도 권한 없이 공개 접근"을
요구하므로 반드시 바꿔야 합니다.

1. 브라우저에서 <https://github.com/DONGJUN92?tab=packages> 접속
2. 목록에서 **`ossp-router`** 클릭
3. 오른쪽 **`Package settings`** 클릭
4. 화면 맨 아래 **`Danger Zone`** 까지 스크롤
5. **`Change visibility`** → **`Public`** 선택 → 확인란에 `ossp-router` 를 그대로 입력 → 확인
6. 같은 화면 위쪽 **`Manage Actions access`** 에서 저장소
   `ossp-2026-llm-router-challenge_shin` 을 연결해 두면 패키지 페이지에 코드가 같이 보입니다

**공개됐는지 확인하는 법**: 브라우저 시크릿 모드로
<https://github.com/DONGJUN92/ossp-2026-llm-router-challenge_shin/pkgs/container/ossp-router>
를 열어 로그인 없이 보이면 성공입니다.

---

## 8단계 · 다이제스트 확인하고 제출 파일 만들기

```powershell
docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/dongjun92/ossp-router:submission
```

`ghcr.io/dongjun92/ossp-router@sha256:` 뒤에 64자리가 붙은 한 줄이 나옵니다.
**그 줄 전체를 복사**해서 아래 명령의 따옴표 안에 넣습니다.

```powershell
python tools/make_submission_metadata.py --image-digest 'ghcr.io/dongjun92/ossp-router@sha256:여기에붙여넣기'
```

이어서 형식 검사:

```powershell
python tools/validate_technical_submission.py
```

**정상이면** 통과 메시지가 나오고, 저장소 루트에 `submission-ossp-skt.json` 이 생깁니다.

---

## 9단계 · 그 파일만 따로 커밋

`docs/SUBMISSION.md` 는 **코드 커밋과 JSON 커밋을 분리**하라고 요구합니다.

```powershell
git add submission-ossp-skt.json
git commit -m "add technical submission metadata"
git push origin main
```

마지막으로 커밋 번호를 확인합니다:

```powershell
git rev-parse HEAD
```

나온 40자리를 넣은 아래 주소가 **결과보고서의 `프로젝트 등록 URL`** 입니다:

```
https://github.com/DONGJUN92/ossp-2026-llm-router-challenge_shin/tree/<40자리>
```

---

## 자주 나오는 문제

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `docker: command not found` | Docker Desktop 이 안 켜짐 | 0단계 |
| `Cannot connect to the Docker daemon` | 같음 | 0단계, 초록불 확인 |
| `unauthorized` (pull) | 토큰 만료 | 4단계 |
| `denied` (push) | 토큰에 `write:packages` 없음 | 4단계 다시, 권한 체크 |
| `exec format error` | arm64 에뮬레이터 미등록 | `docker run --privileged --rm tonistiigi/binfmt --install arm64` |
| 빌드가 `amd64` 로 나옴 | `--platform` 누락 | 5단계 명령 그대로 복사 |
| 패키지가 안 보임 | 비공개 상태 | 7단계 |

## 최종 제출 때 다시 할 것

코드가 바뀌면 **5~9단계만** 반복하면 됩니다. 0~4단계(로그인·토큰)는 한 번만 하면
토큰 만료 전까지 유지됩니다.

# Docker 실행 가이드

UltraBreak를 GPU 컨테이너에서 재현 가능하게 실행하기 위한 환경입니다.
**대상 GPU: NVIDIA H100 (Hopper, sm_90 / CUDA 12.x).**

베이스 이미지 `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`에 torch 2.5.1 +
torchvision 0.20.1이 이미 포함되어 있어 `requirements.txt`와 정확히 일치하며,
그 위에 `transformers>=4.57`(Qwen3-VL용)과 나머지 의존성만 설치합니다.

---

## 1. 사전 요구사항 (호스트)

- NVIDIA 드라이버 (CUDA 12.x 지원, H100)
- Docker Engine 24+ 및 Docker Compose v2+
- **NVIDIA Container Toolkit** (컨테이너에서 GPU 접근)

```bash
# GPU가 컨테이너에서 보이는지 확인
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. 이미지 빌드

```bash
docker compose build
```

---

## 3. 실행

`docker compose run`으로 스크립트를 골라 실행합니다. 레포 전체가
`/app`에 bind-mount 되므로 코드/설정 수정은 **재빌드 없이** 즉시 반영되고,
`outputs/`·`results/`도 호스트에 바로 기록됩니다.

```bash
# (1) 적대적 이미지 최적화 — surrogate: Qwen3-VL-8B-Instruct
docker compose run --rm ultrabreak python optimisation/optimise.py

# (2) 공격(타깃 모델 추론)
docker compose run --rm ultrabreak python evaluation/attack.py

# (3) 평가(HarmBench 등으로 ASR 산출)
docker compose run --rm ultrabreak python evaluation/evaluate.py
```

`docker compose run` 없이 기본 커맨드(`optimisation/optimise.py`)로 바로 띄우려면:

```bash
docker compose up        # 포그라운드
docker compose up -d     # 백그라운드 (로그: docker compose logs -f)
```

컨테이너 안에서 직접 셸을 열어 디버깅:

```bash
docker compose run --rm ultrabreak bash
```

---

## 4. GPU 선택 (공용 서버 필수)

여러 명이 쓰는 서버라면 **할당받은 GPU만** 컨테이너에 노출해야 합니다.
`NVIDIA_VISIBLE_DEVICES`에 **호스트 기준 인덱스**(`nvidia-smi`/`ndtop`에서 보이는 번호)를
넘기면 NVIDIA Container Toolkit이 해당 GPU만 컨테이너에 마운트합니다.

```bash
# 예: 4,5,6,7번을 할당받은 경우
NVIDIA_VISIBLE_DEVICES=4,5,6,7 docker compose run --rm ultrabreak \
    python optimisation/optimise.py
```

매번 입력하기 번거로우면 레포 루트 `.env`에 한 줄 넣어두면 됩니다
(compose가 자동으로 읽음):

```bash
echo 'NVIDIA_VISIBLE_DEVICES=4,5,6,7' >> .env
docker compose run --rm ultrabreak python optimisation/optimise.py
```

> **중요**
> - 기본값은 `all`이라 **그냥 실행하면 8장 전부**를 잡습니다. 공용 서버에서는
>   반드시 `NVIDIA_VISIBLE_DEVICES`를 지정하세요.
> - 컨테이너 **안에서는 GPU가 0부터 재번호**됩니다. 즉 호스트 4,5,6,7 →
>   컨테이너 `cuda:0,1,2,3`. 코드에서 `cuda:0`은 곧 호스트 4번을 가리킵니다.
>   (따라서 컨테이너 내부에서 `CUDA_VISIBLE_DEVICES`를 또 줄 때는 0~3 기준)
> - `nvidia-smi`로 확인: `docker compose run --rm ultrabreak nvidia-smi` →
>   할당받은 4장만 보여야 정상입니다.

---

## 5. 모델 가중치 캐시

Qwen3-VL-8B 등 가중치(수~수십 GB)는 명명된 볼륨 `hf-cache`
(`/cache/huggingface`)에 저장되어 **재실행 시 다시 다운로드하지 않습니다.**

```bash
docker volume inspect ultrabreak_hf-cache   # 위치 확인
docker volume rm ultrabreak_hf-cache        # 캐시 비우기(전체 재다운로드됨)
```

> 첫 실행은 가중치 다운로드로 시간이 걸립니다. 사내 미러나 토큰이 필요하면
> 아래 `.env`의 `HF_TOKEN`을 설정하세요.

---

## 6. 환경변수 / API 키 (선택)

**open-weight VLM(Qwen3-VL, LLaVA, GLM 등)만 쓸 거면 불필요합니다.**
GPT·Claude·Gemini 같은 closed-source 타깃을 평가할 때만
(`evaluation/evaluate.py`가 `python-dotenv` 사용) `.env`가 필요합니다.

```bash
cp .env.example .env
# .env 편집하여 필요한 키만 채우기 (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / HF_TOKEN)
```

`.env`는 `.gitignore`에 포함되어 커밋되지 않습니다. compose는 `.env`가 없어도
정상 동작합니다(`required: false`).

---

## 7. 데이터 / 출력 경로

bind-mount(`./:/app`)이므로 호스트와 컨테이너가 동일 경로를 공유합니다.

| 호스트 | 컨테이너 | 용도 |
|---|---|---|
| `./train_configs` | `/app/train_configs` | 최적화 학습 타깃 설정 |
| `./attack_configs` | `/app/attack_configs` | 공격 쿼리 설정 |
| `./datasets` | `/app/datasets` | 벤치마크 데이터 |
| `./outputs` | `/app/outputs` | 최적화된 패치/이미지·로그 |
| `./results` | `/app/results` | 공격·평가 결과 CSV |
| `hf-cache`(볼륨) | `/cache/huggingface` | 모델 가중치 캐시 |

> 스크립트는 레포 루트를 작업 디렉터리로 가정합니다(예: `./train_configs/test.csv`).
> compose는 `working_dir`가 `/app`이라 그대로 동작합니다.

---

## 8. 트러블슈팅

- **`could not select device driver "nvidia"`** → NVIDIA Container Toolkit 미설치/미설정.
  `nvidia-ctk runtime configure` 후 Docker 재시작.
- **OOM / `CUDA out of memory`** → GPU 수를 늘리거나 `CUDA_VISIBLE_DEVICES`로
  여유 있는 GPU 지정. H100 80GB 단일 카드로 7~8B surrogate 최적화는 가능합니다.
- **`Qwen3VLForConditionalGeneration`를 못 찾음** → 이미지의 transformers가 4.57 미만.
  `docker compose build --no-cache`로 재빌드.
- **dataloader 관련 공유메모리 에러** → compose에 `shm_size: 16gb`, `ipc: host`가
  이미 설정되어 있습니다. 더 필요하면 값을 올리세요.

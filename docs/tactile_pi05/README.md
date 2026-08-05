# pi0.5 SO101 Paxini Tactile Rollout

이 문서는 학습이 끝난 pi0.5 PEFT adapter를 VTLA에서 복원해서 SO101 +
Paxini tactile robot으로 rollout하는 최소 절차입니다.

## 필요한 것

- VTLA 코드: 이 repo revision이 필요합니다. A15/A2 adapter는 로컬 pi0.5
  tactile injection 코드에 의존합니다.
- Base pi0.5: 기본 경로는 `/workspace/vtla/hf_models/pi05_base`입니다.
- Adapter: `pretrained_model/` 폴더 전체가 필요합니다.
- Robot wrapper: tactile rollout은 `paxini_so101_follower`를 씁니다.

`adapter_model.safetensors` 하나만 복사하면 부족합니다. 아래 파일들이 들어있는
`pretrained_model/` 폴더를 통째로 사용하세요.

```text
adapter_config.json
adapter_model.safetensors
config.json
policy_preprocessor.json
policy_postprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

## 설치

```bash
cd /workspace/VTLA
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[pi,core_scripts,paxini]"
```

캐시 경로는 RunPod/서버에서 유지되는 위치로 맞춥니다.

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_LEROBOT_HOME=/workspace/.cache/lerobot
export TORCH_HOME=/workspace/vtla/torch_cache
export TOKENIZERS_PARALLELISM=false
```

## Base pi0.5 준비

adapter config는 기본적으로 아래 경로를 기대합니다.

```bash
/workspace/vtla/hf_models/pi05_base
```

없으면 다운로드합니다.

```bash
mkdir -p /workspace/vtla/hf_models
hf download lerobot/pi05_base \
  --local-dir /workspace/vtla/hf_models/pi05_base
```

다른 위치에 이미 있다면 symlink를 둡니다.

```bash
mkdir -p /workspace/vtla/hf_models
ln -s /actual/path/to/pi05_base /workspace/vtla/hf_models/pi05_base
```

PaliGemma tokenizer local path가 필요하면 helper로 패치합니다.

```bash
export HF_TOKEN=<token-with-paligemma-access>
bash experiments/pi05_cup_left/scripts/prepare_pi05_assets.sh
bash experiments/pi05_cup_left/scripts/preflight_pi05.sh
```

## Adapter 복원

Hugging Face backup repo에서 adapter를 받습니다. 실제 repo id가 다르면
`ADAPTER_REPO`만 바꾸세요.

```bash
export ADAPTER_REPO=Leedohyeon/pi05-so101-paxini-cup-left-adapters

rm -rf /workspace/vtla/hf_restore/pi05_cup_left_adapters
hf download "$ADAPTER_REPO" \
  --local-dir /workspace/vtla/hf_restore/pi05_cup_left_adapters
```

권장 rollout 후보:

```text
A15_cnn_tactile_prefix_r16/checkpoints/020000/pretrained_model
A15_cnn_tactile_prefix/checkpoints/020000/pretrained_model
A2_gated_action_tactile/checkpoints/020000/pretrained_model
A0_no_tactile/checkpoints/020000/pretrained_model
```

A15/A2는 tactile 입력을 쓰는 adapter입니다. A0는 no-tactile baseline입니다.

## Rollout

A15 r=16 20k 후보:

```bash
source .venv/bin/activate

lerobot-rollout \
  --policy.path=/workspace/vtla/hf_restore/pi05_cup_left_adapters/A15_cnn_tactile_prefix_r16/checkpoints/020000/pretrained_model \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --fps=30
```

A2 후보:

```bash
lerobot-rollout \
  --policy.path=/workspace/vtla/hf_restore/pi05_cup_left_adapters/A2_gated_action_tactile/checkpoints/020000/pretrained_model \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --fps=30
```

`paxini_so101_follower`는 `observation.tactile.primary`를 `(2, 9, 9)` 형태로
내보냅니다. A15/A2 adapter는 이 feature shape에 맞춰 학습되었습니다.

## Offline action comparison

실제 로봇에 명령을 보내기 전에, 학습 episode의 observation으로 A0/A15/A2의
예측 action과 demonstration action을 비교합니다. 먼저 적은 frame으로 smoke
test를 실행합니다.

RunPod에 dataset extra와 FFmpeg가 필요합니다.

```bash
uv sync --locked --python 3.12 --extra dataset --extra async --extra pi --extra peft
apt-get update && apt-get install -y ffmpeg
```

```bash
EPISODE=0 STRIDE=10 MAX_FRAMES=10 bash user_scripts/offline_eval.sh
```

성공 episode 번호를 확인한 뒤 전체 frame을 평가합니다.

```bash
EPISODE=0 STRIDE=1 MAX_FRAMES=0 bash user_scripts/offline_eval.sh
```

동일 observation의 sampling seed 분산과 teacher-forced loss를 함께 검사하려면:

```bash
EPISODE=0 STRIDE=20 NUM_SEEDS=8 TEACHER_LOSS_REPEATS=8 \
  bash user_scripts/offline_eval.sh
```

결과는 기본적으로
`/workspace/vtla/outputs/offline_eval/episode_<N>_<timestamp>/`에 저장됩니다.
각 모델 폴더에는 summary JSON, 전체 action chunk NPZ, 첫 action CSV와 비교 plot이
생성됩니다. 이 검사는 robot을 생성하거나 motor command를 보내지 않습니다.

## 자주 나는 문제

- `paxini_so101_follower`를 못 찾으면 `pip install -e ".[pi,core_scripts,paxini]"`
  설치가 빠졌거나 다른 repo의 venv를 쓰고 있는 것입니다.
- adapter 로딩 중 base model 경로 오류가 나면
  `/workspace/vtla/hf_models/pi05_base`가 실제 base model을 가리키는지 확인하세요.
- tactile 없는 상태에서 A15/A2를 돌리면 실행은 될 수 있지만 tactile test가 아닙니다.

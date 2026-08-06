# pi0.5 SO101 Paxini Cup Experiments

SO101 + Paxini cup-left dataset으로 pi0.5 PEFT adapter를 학습/재개하는
wrapper입니다. 실제 robot rollout만 할 때는
`docs/tactile_pi05/README.md`를 먼저 보세요.

## 기본 경로

```bash
REPO_DIR=/root/vtla/workspace/VTLA
DATASET_ROOT=/workspace/vtla/data/paxini_cup_left_60
RUN_ROOT=/workspace/vtla/outputs/pi05_cup_left
PI05_BASE_PATH=/workspace/vtla/hf_models/pi05_base
TOKENIZER_DIR=/workspace/vtla/hf_models/paligemma-3b-pt-224-tokenizer
```

dataset feature:

- `observation.images.belly`
- `observation.images.wrist`
- `observation.images.top`
- `observation.tactile.primary` with shape `[2, 9, 9]`
- `observation.state`
- `action`

학습 script는 camera key를 pi0.5가 기대하는 이름으로 rename합니다.

```json
{
  "observation.images.belly": "observation.images.base_0_rgb",
  "observation.images.wrist": "observation.images.left_wrist_0_rgb",
  "observation.images.top": "observation.images.right_wrist_0_rgb"
}
```

## 방법

- `A0`: no tactile baseline, `policy.tactile_injection_mode=none`
- `A1.5`: CNN tactile prefix token, `policy.tactile_injection_mode=prefix`
- `A2`: gated action-side tactile tokens, `policy.tactile_injection_mode=action_gated`

공통 설정:

- `freeze_vision_encoder=true`
- `train_expert_only=true`
- `dtype=bfloat16`
- `peft.method=LORA`
- `peft.r=8`

## base/tokenizer 준비

`lerobot/pi05_base`와 PaliGemma tokenizer local path를 준비합니다.

```bash
export HF_TOKEN=<token-with-paligemma-access>
bash experiments/pi05_cup_left/scripts/prepare_pi05_assets.sh
bash experiments/pi05_cup_left/scripts/preflight_pi05.sh
```

## smoke

한 step씩 먼저 확인합니다.

```bash
bash experiments/pi05_cup_left/scripts/run_pi05_experiment.sh A0 1
bash experiments/pi05_cup_left/scripts/run_pi05_experiment.sh A1.5 1
bash experiments/pi05_cup_left/scripts/run_pi05_experiment.sh A2 1
```

짧은 queue:

```bash
bash experiments/pi05_cup_left/scripts/queue_smoke.sh 10
```

## 학습

A0, A1.5, A2를 순서대로 실행합니다.

```bash
BATCH_SIZE=8 NUM_WORKERS=4 PERSISTENT_WORKERS=true PREFETCH_FACTOR=2 \
  LOG_FREQ=25 SAVE_FREQ=1000 \
  bash experiments/pi05_cup_left/scripts/queue_main.sh 4000
```

최신 checkpoint에서 target step까지 재개합니다.

```bash
bash experiments/pi05_cup_left/scripts/queue_resume_latest.sh 8000
bash experiments/pi05_cup_left/scripts/queue_resume_latest.sh 20000
```

RunPod에서 VSCode를 닫아도 계속 돌리려면:

```bash
setsid bash -lc "bash experiments/pi05_cup_left/scripts/queue_resume_latest.sh 20000 >> /workspace/vtla/experiments/pi05_cup_left/logs/queue_resume_20000.log 2>&1" &
```

## checkpoint

rollout에는 해당 step의 `pretrained_model/` 폴더 하나와 base pi0.5 model,
tokenizer, 이 코드 revision이 필요합니다.

예시:

```bash
/workspace/vtla/outputs/pi05_cup_left/A0_no_tactile_4000steps_*/checkpoints/020000/pretrained_model
/workspace/vtla/outputs/pi05_cup_left/A1.5_cnn_tactile_prefix_4000steps_*/checkpoints/020000/pretrained_model
/workspace/vtla/outputs/pi05_cup_left/A2_gated_action_tactile_4000steps_*/checkpoints/020000/pretrained_model
```

학습 재개에는 `pretrained_model/`만으로 부족합니다. optimizer/scheduler state가
`training_state/`에 있으므로 numbered checkpoint 전체를 복사하세요.

## rollout

robot rollout machine에는 robot dependency까지 설치합니다.

```bash
pip install -e ".[pi,core_scripts,paxini]"
```

adapter `pretrained_model/`을 `--policy.path`로 지정합니다.

```bash
lerobot-rollout \
  --policy.path=/workspace/vtla/outputs/pi05_cup_left/A2_gated_action_tactile_4000steps_*/checkpoints/020000/pretrained_model \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --fps=30
```

주의: A1.5/A2는 `paxini_so101_follower`가 중요합니다. 이 wrapper가
`observation.tactile.primary`를 `(2, 9, 9)`로 내보내며, adapter도 그 shape에
맞춰 학습되었습니다.

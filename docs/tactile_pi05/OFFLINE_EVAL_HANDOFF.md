# VTLA PI0.5 Offline Evaluation Handoff

작성일: 2026-07-02

## 목적

실제 SO-ARM101 rollout에서 A0, A1.5, A2 모두 컵 앞으로 정상 접근하지 못하고
관절이 급격하게 움직였다. 실제 로봇 시험을 중단하고, 성공 학습 episode의
observation을 체크포인트에 직접 입력해 predicted action과 demonstration action을
비교했다.

이 문서는 현재까지 수행한 검사와 다음 작업을 다른 agent가 바로 이어받을 수 있도록
정리한 handoff다.

## 구성

- 코드: `welry9277/VTLA`
- 데이터셋: `outputs/paxini_cup_left_60`
  - 60 episodes, 32,463 frames, 30 FPS
  - robot type: `paxini_so101_follower`
  - action/state: 6-DoF joint position
  - camera: belly, wrist, top
  - tactile: `observation.tactile.primary`, shape `(2, 9, 9)`
- 정확한 task 문자열:
  - `Pick one cup from the stacked cups and place it on the left side.`
- Base model: `lerobot/pi05_base`
- Adapter repo: `Leedohyeon/pi05-so101-paxini-cup-left-adapters`

### 비교 체크포인트

```text
A0
all_checkpoints/pi05_cup_left/A0_no_tactile_4000steps_20260629_151550/checkpoints/020000/pretrained_model

A1.5
all_checkpoints/pi05_cup_left/A1.5_cnn_tactile_prefix_4000steps_20260629_163919/checkpoints/020000/pretrained_model

A2
all_checkpoints/pi05_cup_left/A2_gated_action_tactile_4000steps_20260629_180545/checkpoints/020000/pretrained_model
```

폴더명의 `4000steps`는 최초 run 이름일 뿐이며 실제 checkpoint는 모두 20k다.

## 구현한 offline evaluator

주요 파일:

- `src/lerobot/scripts/lerobot_offline_action_eval.py`
- `user_scripts/offline_eval.sh`
- `pyproject.toml`의 `lerobot-offline-action-eval` entry point
- `user_scripts/server.sh`의 `PREPARE_ONLY=true` 지원

Evaluator 기능:

- 저장된 checkpoint preprocessor/postprocessor 사용
- episode observation과 50-step demonstration action chunk 로드
- 전체/관절별 chunk MAE, RMSE
- 첫 predicted action MAE
- state 대비 predicted/demo action 이동량
- action 방향 일치율
- episode padding 제외
- tactile shape/min/max/mask 기록
- preprocessing/inference/postprocessing timing
- prediction/demo plot, CSV, NPZ, JSON 저장
- 동일 observation을 여러 random seed로 반복하는 sampling 분산 검사
- demonstration action을 넣은 teacher-forced flow-matching loss 검사
- FFmpeg/video decode를 14 GB base model보다 먼저 검증
- robot 객체를 만들거나 motor command를 전송하지 않음

검증 완료:

- Ruff 통과
- Python compile 통과
- shell syntax 통과
- synthetic metric/padding test 통과
- RunPod H100 실제 model/dataset smoke test 통과

## RunPod 환경 상태

- GPU: NVIDIA H100 80 GB
- repo: `/workspace/VTLA`
- dataset 복사 완료:
  - `/workspace/vtla/data/paxini_cup_left_60`
- base/tokenizer:
  - `/workspace/vtla/hf_models/pi05_base`
  - `/workspace/vtla/hf_models/paligemma-3b-pt-224-tokenizer`
- adapter restore:
  - `/workspace/vtla/hf_restore/all_checkpoints/pi05_cup_left`
- Python 3.12 `.venv` 재생성 완료
- dataset/async/pi/peft extras 설치 완료
- FFmpeg 설치 완료
- evaluator 최신 파일을 RunPod repo에 직접 복사해 둠

주의: RunPod repo에는 이 작업 전부터 사용자가 수정한 async inference 및 PI0.5 파일이
있었다. 해당 변경은 덮어쓰지 않았다.

## 확인된 pipeline 설정

- action representation: absolute joint position
- action shape/order:
  1. `shoulder_pan.pos`
  2. `shoulder_lift.pos`
  3. `elbow_flex.pos`
  4. `wrist_flex.pos`
  5. `wrist_roll.pos`
  6. `gripper.pos`
- state/action joint order 동일
- delta action processor: disabled
- absolute action postprocessor: disabled
- normalization: state/action `QUANTILES`
- visual/tactile normalization: `IDENTITY`
- checkpoint postprocessor action stats와 dataset `meta/stats.json` 값이 정확히 일치
- camera rename:
  - belly → `base_0_rgb`
  - wrist → `left_wrist_0_rgb`
  - top → `right_wrist_0_rgb`
- task prompt는 dataset의 정확한 문자열을 사용
- A1.5/A2 tactile mask는 `True`
- tactile raw shape는 `(2, 9, 9)`

### Tactile adapter weight 확인

Base PI0.5에는 tactile module이 없으므로 base load 때 missing-key 경고가 발생한다.
그 뒤 adapter가 tactile weight를 채운다.

- A1.5 adapter: tactile 관련 weight 25개 포함
- A2 adapter: tactile 관련 weight 8개 포함

따라서 현재 missing-key 로그는 base load 단계에서 예상되는 경고이며 tactile weight
누락을 의미하지 않는다.

## 검사 1: Episode 0 전체 408-frame 비교

한 seed로 episode 0의 모든 frame을 평가했다.

| 모델 | Chunk MAE | First-action MAE | 방향 일치율 | 예측 평균 점프 | 예측 최대 점프 | Demo 평균 점프 |
| ---- | --------: | ---------------: | ----------: | -------------: | -------------: | -------------: |
| A0   |     9.094 |            8.832 |       55.2% |          9.115 |         98.961 |          1.583 |
| A1.5 |     8.420 |            7.847 |       56.5% |          8.183 |         93.285 |          1.583 |
| A2   |     8.915 |            8.355 |       55.2% |          8.646 |         96.441 |          1.583 |

관절 position native unit 기준이다. 회전 관절은 사실상 degree이고 gripper는 별도
position scale이다.

결과 위치:

- local: `outputs/offline_eval_episode_0_full/`
- RunPod: `/workspace/vtla/outputs/offline_eval/episode_0_full/`

### 관찰

- 세 모델 모두 demonstration보다 약 5배 큰 첫 action 이동을 생성한다.
- 최대 점프는 약 93–99로 실제 rollout에서 관찰된 급격한 움직임과 일치한다.
- A1.5가 상대적으로 가장 낫지만 실기 실행에는 여전히 위험하다.
- 세 모델이 유사하게 실패하므로 tactile fusion만의 문제일 가능성은 낮다.

### 궤적 상관도와 lag

주요 관절은 predicted/demo 상관도가 약 0.72–0.84로, 큰 동작 단계 자체는 어느 정도
학습했다. 최적 lag는 주로 2–10 frames였고 상관도 개선은 작았다. 따라서 단순한
action time shift만으로 큰 오차와 점프를 설명할 수 없다.

## 검사 2: Seed 분산 + teacher-forced loss

Episode 0에서 20-frame 간격으로 21개 observation을 선택했다.

- observation당 action sampling seed 8회
- observation당 teacher-forced forward 8회
- 총 모델별 336 forward

| 모델 | Teacher loss 평균 | Seed별 first-action MAE 평균 | First-action seed 표준편차 | 평균 seed 범위 | 최대 seed 범위 |
| ---- | ----------------: | ---------------------------: | -------------------------: | -------------: | -------------: |
| A0   |            0.1466 |                         9.08 |                      7.747 |         24.376 |         79.695 |
| A1.5 |            0.1376 |                         7.96 |                      6.526 |         20.524 |         62.780 |
| A2   |            0.1486 |                         8.62 |                      7.116 |         22.416 |         67.838 |

Teacher loss와 sampled first-action MAE의 frame별 상관도:

- A0: 0.077
- A1.5: 0.039
- A2: 0.055

8개 seed action을 평균했을 때 first-action MAE:

- A0: 6.686
- A1.5: 5.952
- A2: 6.623

평균을 내도 demo 평균 점프 1.715에 비해 predicted 평균 점프는 6.3–6.9로 크다.

결과 위치:

- local: `outputs/offline_eval_episode_0_stochastic_diagnostics/`
- RunPod: `/workspace/vtla/outputs/offline_eval/episode_0_stochastic_diagnostics/`

## 현재 판정

### 가능성이 낮아진 원인

- 잘못된 checkpoint 경로
- adapter weight 누락
- normalization stats 불일치
- preprocessor/postprocessor 누락
- absolute/delta action representation 혼동
- joint ordering 불일치
- camera key mapping 불일치
- task prompt 불일치
- tactile shape/mask 누락
- 단순한 action time shift

### 가장 강한 현재 가설

모델은 학습 observation에서 큰 궤적을 어느 정도 학습했고 teacher-forced loss도 학습
로그와 비슷하게 낮다. 그러나 동일 observation의 sampled action이 seed에 따라 매우
크게 달라진다.

즉 현재 직접적인 실패 원인은 다음에 가깝다.

> Flow-matching training objective는 낮아졌지만, inference sampling으로 얻는 action
> distribution이 지나치게 넓고 첫 action이 불안정하다.

이는 아래 중 하나일 수 있다.

1. `num_inference_steps=10`의 ODE integration이 부족함
2. 60-episode 데이터로 conditional action distribution을 충분히 좁히지 못함
3. 학습 설정(`train_expert_only`, LoRA rank 8, frozen vision)이 task에 부족함
4. flow-matching loss가 실제 first-action 품질을 잘 반영하지 못함
5. dataset의 observation-action conditional ambiguity 또는 demonstration 품질 문제

현재 세 config 공통값:

```text
num_inference_steps = 10
chunk_size = 50
n_action_steps = 50
dtype = bfloat16
gradient_checkpointing = true
train_expert_only = true
freeze_vision_encoder = true
```

## 다음 작업 우선순위

### 1. Inference step sensitivity

가장 먼저 동일한 21 frames × 8 seeds에서 `num_steps=10, 20, 50`을 비교한다.
`predict_action_chunk(processed, num_steps=N)`으로 전달할 수 있다.

기록할 값:

- seed별 first-action 표준편차
- seed range
- first-action MAE
- state 대비 predicted jump
- inference latency

판정:

- step을 늘렸을 때 분산/MAE가 크게 감소하면 sampler integration 문제
- 거의 그대로면 학습된 action distribution 자체가 넓은 문제

### 2. Checkpoint sweep

동일 evaluator로 4k, 8k, 12k, 20k checkpoint를 비교한다.

- teacher loss가 감소하면서 sampled MAE가 악화됐는지 확인
- 20k가 과적합/불안정해졌는지 확인
- training loss만 checkpoint 선택 기준으로 쓰지 않기

### 3. Sampling noise와 output 분석

- 동일 observation의 normalized action을 postprocessor 전/후로 저장
- 관절별 seed variance 확인
- noise scale 또는 deterministic noise 실험
- seed 고정은 재현성만 주며 정확도 해결책은 아님
- multi-seed 평균은 진단용이며 실시간 rollout 해결책으로 간주하지 않음

### 4. Dataset/action quality 검사

- 동일하거나 매우 유사한 observation에 서로 다른 action이 있는지 nearest-neighbor 검사
- state → first demonstration action delta 분포와 outlier 확인
- episode 단계별 접근/접촉/파지/분리/이동/배치 annotation 추가
- 단계별 teacher loss와 sampled MAE 비교

### 5. 이후 안전 순서

1. 원인이 설명될 때까지 full robot rollout 금지
2. live observation dry-run으로 motor command 없이 prediction/state 기록
3. 학습 이미지와 live processor 최종 이미지 비교
4. 관절별 relative target 제한 적용
5. 최초 실기는 1–2 action만 실행
6. current state 대비 action jump가 임계값을 넘으면 실행 차단

## 재실행 예시

기본 offline 평가:

```bash
EPISODE=0 STRIDE=1 MAX_FRAMES=0 bash user_scripts/offline_eval.sh
```

Seed/teacher-loss 진단:

```bash
EPISODE=0 STRIDE=20 NUM_SEEDS=8 TEACHER_LOSS_REPEATS=8 \
  bash user_scripts/offline_eval.sh
```

RunPod 필수 환경:

```bash
/root/.local/bin/uv sync --locked --python 3.12 \
  --extra dataset --extra async --extra pi --extra peft
apt-get update && apt-get install -y ffmpeg
```

## 안전 결론

현재 모델들은 학습 episode에서도 first action이 demonstration과 크게 다르고 seed
분산이 매우 크다. 원인이 해소되기 전에는 실제 로봇 full rollout을 재개하지 않는다.

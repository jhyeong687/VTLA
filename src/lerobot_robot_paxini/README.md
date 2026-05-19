# LeRobot + Paxini 촉각 센서 사용 가이드

이 문서는 Paxini GEN3 촉각 센서를 LeRobot SO101 follower에 붙이고, 카메라/로봇 상태/action/촉각 데이터를 기록한 뒤 Pi0.5 계열 policy 학습까지 연결하는 흐름을 정리한 것입니다.

더 자세한 구조 설명과 공식 LeRobot 대비 변경점은 `PAXINI_PI05_GUIDE.ko.md`를 보십시오. 같은 내용의 보기용 파일로 `PAXINI_PI05_GUIDE.ko.html`과 `PAXINI_PI05_GUIDE.ko.pdf`도 생성해두었습니다.

## 현재 구성

LeRobot에 연결하기 위해 쓰는 파일만 남겼습니다.

- `paxini_so101.py`: 기존 SO101 follower에 촉각 observation을 추가한 robot class입니다.
- `config_paxini_so101.py`: serial port, tactile image size, taxel 수, CSV replay 설정을 담는 config입니다.
- `paxini_reader.py`: 실제 serial 센서, mock 센서, PXSR CSV replay를 읽는 reader입니다.
- `tactile_render.py`: 154개 taxel 값을 `(64, 64)` heatmap 또는 `(2, 9, 9)` stacked grid로 바꿉니다.
- `types.py`: Paxini reader가 반환하는 sample dataclass입니다.
- `make_pi05_smoke_dataset.py`: 실제 로봇/센서 없이 CSV나 mock으로 LeRobot dataset을 만드는 smoke test 스크립트입니다.

제조사 GUI/CLI 예제는 프로토콜 확인에는 유용했지만 record/train 런타임에는 쓰지 않기 때문에 이 패키지에서 제거했습니다.

## 센서 데이터 구조

현재 확인된 연결은 ring proximal + pinky proximal 두 개입니다.

- Ring proximal: 77개 taxel
- Pinky proximal: 77개 taxel
- 합계: 154개 Z force 값

PXSR CSV에서는 보통 이런 컬럼으로 저장됩니다.

```text
3-0-NxN-Z[0] ... 3-0-NxN-Z[76]
4-0-NxN-Z[0] ... 4-0-NxN-Z[76]
```

여기서는 CSV 컬럼 순서대로 `3-0` 77개와 `4-0` 77개를 이어 붙여서 총 154개 tactile vector로 사용합니다.

현재 지원하는 tactile 표현은 두 가지입니다.

```text
heatmap       : 154개 taxel -> (64, 64)
stacked_9x9   : 154개 taxel -> (2, 9, 9)
```

추천은 `stacked_9x9`입니다. 각 sensor의 77개 taxel을 9x9 격자에 넣고 네 모서리만 0으로 채웁니다. 그러면 ring과 pinky가 각각 한 채널이 되어 `observation.tactile.primary`의 shape가 `(2, 9, 9)`가 됩니다. 기본값은 frame-wise normalize를 하지 않고 센서 값에 `tactile_taxel_scale`만 곱한 값을 저장합니다. Pi0.5 tactile CNN은 이 값을 `Conv2d(in_channels=2, kernel_size=3)` 형태로 직접 처리할 수 있습니다.

LeRobot dataset에 들어가는 주요 feature는 다음과 같습니다.

```text
observation.state              # SO101 follower joint position 6개
observation.images.front        # 카메라 이미지, 있으면 저장
observation.tactile.primary     # 촉각, shape=(64, 64) 또는 (2, 9, 9)
action                          # leader가 만든 follower target action 6개
task                            # 자연어 task 문장
timestamp, frame_index, episode_index ...
```

## 기본 환경

repo 루트에서 실행합니다.

```bash
cd /mnt/c/Users/dlehg/conference/vtla
conda activate lerobot
export PYTHONPATH=/mnt/c/Users/dlehg/conference/vtla/lerobot/src:$PYTHONPATH
pip install "paxini-utils @ git+https://github.com/riverfog7/paxini-utils.git"
```

`paxini-utils`는 Paxini high-speed board의 register command, connected sensor 확인, point count, distribution force 파싱을 담당합니다. LeRobot 쪽에서는 `read_connected_distribution_forces()`를 호출해서 연결된 센서의 촉각 데이터를 요청하고, serial frame은 직접 파싱하지 않습니다.

WSL에서 USB serial을 쓸 때는 Windows COM 포트를 WSL로 attach해야 합니다. Windows에서 PXSR 프로그램이 COM 포트를 잡고 있으면 WSL에서는 동시에 읽을 수 없습니다. 실제 센서를 쓸 때는 PXSR 프로그램을 종료하고 `/dev/ttyACM*` 포트를 확인합니다.

```bash
ls -l /dev/serial/by-id/
python -m serial.tools.list_ports -v
```

## 센서 없이 CSV로 파이프라인 확인

PXSR에서 저장한 CSV가 있으면 실제 촉각 센서 없이도 LeRobot dataset을 만들 수 있습니다.

예시 CSV:

```text
/mnt/c/Users/dlehg/AppData/Roaming/pxsr-gen3/DataLogging/2026-05-16-182429.csv
```

CSV로 smoke dataset 만들기:

```bash
conda run -n lerobot python -m lerobot_robot_paxini.make_pi05_smoke_dataset \
  --repo-id local/paxini_pi05_smoke_csv \
  --root /tmp/lerobot_paxini_smoke \
  --csv /mnt/c/Users/dlehg/AppData/Roaming/pxsr-gen3/DataLogging/2026-05-16-182429.csv \
  --frames 32 \
  --fps 10
```

`(2, 9, 9)` stacked grid로 만들려면 아래 옵션을 추가합니다.

```bash
conda run -n lerobot python -m lerobot_robot_paxini.make_pi05_smoke_dataset \
  --repo-id local/paxini_pi05_smoke_2x9x9 \
  --root /tmp/lerobot_paxini_smoke_2x9x9 \
  --csv /mnt/c/Users/dlehg/AppData/Roaming/pxsr-gen3/DataLogging/2026-05-16-182429.csv \
  --frames 32 \
  --fps 10 \
  --tactile-representation stacked_9x9
```

생성된 dataset 확인:

```bash
conda run -n lerobot lerobot-info \
  --repo-id local/paxini_pi05_smoke_csv \
  --root /tmp/lerobot_paxini_smoke
```

Python에서 shape 확인:

```bash
conda run -n lerobot python -c "\
from lerobot.datasets import LeRobotDataset; \
ds=LeRobotDataset('local/paxini_pi05_smoke_csv', root='/tmp/lerobot_paxini_smoke'); \
item=ds[0]; \
print(len(ds)); \
print(item['observation.state'].shape); \
print(item['observation.tactile.primary'].shape); \
print(item['observation.images.front'].shape); \
print(item['action'].shape)"
```

`heatmap`이면 대략 아래처럼 나옵니다.

```text
32
torch.Size([6])
torch.Size([64, 64])
torch.Size([3, 224, 224])
torch.Size([6])
```

`stacked_9x9`이면 tactile shape만 아래처럼 바뀝니다.

```text
torch.Size([2, 9, 9])
```

## 실제 센서로 teleoperate 확인

먼저 leader와 follower, tactile port를 각각 확인합니다.

```bash
lerobot-find-port
ls -l /dev/serial/by-id/
```

그 다음 follower가 움직이고 tactile observation까지 붙는지 확인합니다.

```bash
lerobot-teleoperate \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM_LEADER \
  --teleop.id=so101_leader \
  --fps=30 \
  --display_data=true
```

센서 없이 코드 경로만 확인하려면 mock을 켤 수 있습니다.

```bash
lerobot-teleoperate \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_mock=true \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM_LEADER \
  --teleop.id=so101_leader \
  --fps=30
```

CSV를 실제 센서 대신 replay하고 싶으면 아래처럼 씁니다.

```bash
lerobot-teleoperate \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_csv_path=/mnt/c/Users/dlehg/AppData/Roaming/pxsr-gen3/DataLogging/2026-05-16-182429.csv \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM_LEADER \
  --teleop.id=so101_leader \
  --fps=30
```

실제 serial 센서는 기본적으로 background thread에서 `read_connected_distribution_forces()`를 반복 호출해 계속 읽습니다.
record loop는 최신 sample만 가져가므로 촉각 register 요청 대기 때문에 전체 FPS가 흔들리는 일을 줄입니다.
문제 추적을 위해 동기식으로 되돌리고 싶을 때만 아래 옵션을 추가합니다.

```bash
--robot.tactile_async_read=false
```

## 데이터 기록: lerobot-record

record는 매 프레임 아래 순서로 실행됩니다.

```text
robot.get_observation()
teleop.get_action()
robot.send_action()
dataset.add_frame()
```

즉 저장되는 것은 `현재 관측 + 사람이 leader로 준 action`입니다.

실제 센서와 카메라를 쓰는 예:

```bash
lerobot-record \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM_LEADER \
  --teleop.id=so101_leader \
  --dataset.repo_id=dlehg/so101_paxini_ring_pinky \
  --dataset.single_task="pick up the object using ring and pinky tactile feedback" \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false
```

처음에는 반드시 `num_episodes=1` 또는 `5`처럼 작게 시작합니다. 기록 후 dataset shape와 replay가 정상인지 확인한 뒤 긴 데이터 수집으로 넘어가는 것이 안전합니다.

## 기록 재생: lerobot-replay

replay는 학습이 아닙니다. 저장된 `action`만 다시 로봇에 보내서 기록이 제대로 되었는지 확인하는 단계입니다.

```bash
lerobot-replay \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --dataset.repo_id=dlehg/so101_paxini_ring_pinky_YYYYMMDD_HHMMSS \
  --dataset.episode=0
```

replay가 이상하면 학습 전에 record 데이터가 잘못된 것입니다.

## Pi0.5 학습

LeRobot에서 Pi0.5 policy 이름은 `pi05`입니다.

HF base checkpoint는 다음입니다.

```text
lerobot/pi05_base
```

주의할 점:

- `lerobot/pi05_base`의 weight는 약 14.47GB입니다.
- Pi0.5는 4B급 모델이라 8GB GPU에서는 full fine-tune이 어렵습니다.

작은 smoke train:

`stacked_9x9` dataset을 쓰면 Pi0.5 tactile encoder는 `input_shape=(2, 9, 9)`를 보고 첫 convolution을 `in_channels=2`로 만듭니다. 즉 ring/pinky 두 채널이 처음 CNN layer에서 같이 섞입니다.

```bash
lerobot-train \
  --dataset.repo_id=local/paxini_pi05_smoke_2x9x9 \
  --dataset.root=/tmp/lerobot_paxini_smoke_2x9x9 \
  --policy.type=pi05 \
  --policy.paligemma_variant=gemma_300m \
  --policy.action_expert_variant=gemma_300m \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.tactile_encoder_type=cnn \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --steps=1 \
  --num_workers=0 \
  --eval_freq=0 \
  --save_freq=1 \
  --wandb.enable=false
```

실제 fine-tune은 base checkpoint를 지정합니다.

```bash
lerobot-train \
  --dataset.repo_id=dlehg/so101_paxini_ring_pinky_YYYYMMDD_HHMMSS \
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.tactile_encoder_type=cnn \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --steps=3000 \
  --save_freq=1000 \
  --eval_freq=0 \
  --wandb.enable=false
```

## 학습된 policy 실행: lerobot-rollout

학습 후 실제 로봇에서 policy를 실행할 때는 replay가 아니라 rollout을 씁니다.

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/YYYY-MM-DD/HH-MM-SS_pi05/checkpoints/last/pretrained_model \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM_FOLLOWER \
  --robot.id=so101_paxini \
  --robot.tactile_port=/dev/paxini_tactile \
  --robot.tactile_num_taxels=154 \
  --robot.tactile_representation=stacked_9x9 \
  --task="pick up the object using ring and pinky tactile feedback" \
  --duration=60
```

## 자주 막히는 부분

### WSL에서 `/dev/ttyACM*`가 안 보임

Windows에서 COM 포트를 WSL에 attach해야 합니다. PXSR 프로그램이 COM 포트를 잡고 있으면 WSL에서 동시에 읽을 수 없습니다.

### `paxini_so101_follower`를 못 찾음

`PYTHONPATH`가 repo source를 가리키는지 확인합니다.

```bash
export PYTHONPATH=/mnt/c/Users/dlehg/conference/vtla/lerobot/src:$PYTHONPATH
```

### tactile shape가 이상함

현재 기준은 ring+pinky proximal만 사용하므로 `154`개가 맞습니다.

```bash
--robot.tactile_num_taxels=154
```

`158`이 나오면 total force byte를 distributed taxel처럼 잘못 세고 있을 가능성이 큽니다.

## 추천 작업 순서

1. CSV smoke dataset 생성
2. dataset feature shape 확인
3. Pi0.5 smoke train 1 step
4. 실제 센서로 `lerobot-teleoperate`
5. `lerobot-record`로 1 episode만 기록
6. `lerobot-replay`로 action 검증
7. 충분한 episode 기록
8. `lerobot-train`으로 Pi0.5 fine-tune
9. `lerobot-rollout`으로 실제 실행

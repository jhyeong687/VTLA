# VTLA 아키텍처 (Vision-Tactile-Language-Action)

이 문서는 LeRobot 위에 촉각(tactile) 센싱을 확장한 VTLA의 구조를 정리합니다.
Paxini 촉각 센서가 붙은 SO-101 로봇에서 데이터를 얻고, pi0.5 policy에
촉각 정보를 주입해 action을 예측하는 전체 파이프라인을 다룹니다.

## 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HARDWARE                                     │
│   SO-101 arm  +  Paxini 촉각 센서 (154 taxels = M3025 x 2, 77 each)   │
│   카메라: AVFoundation(macOS) / WebRTC(원격) / OpenCV / RealSense     │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ROBOT WRAPPER   src/lerobot_robot_paxini/paxini_so101.py             │
│    PaxiniSO101Follower (SOFollower 확장)                              │
│    ├─ Reader:  SerialPaxiniReader | CsvPaxiniReader | MockPaxiniReader│
│    └─ Render:  taxels_to_stacked_9x9()  또는  TactileHeatmapRenderer  │
│                                                                       │
│  get_observation() →                                                  │
│    observation.state          (관절 상태)                             │
│    observation.images.*       (카메라 프레임)                         │
│    observation.tactile.primary  →  (2, 9, 9)  또는  (64, 64) heatmap  │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING  src/lerobot/processor/tactile_processor.py            │
│    TactileValidationProcessorStep   ("tactile_validation")            │
│       - shape/dtype 검증, numpy → float32 tensor                      │
│    TactileTemporalFilterProcessorStep ("tactile_temporal_filter")     │
│       - EMA(지수이동평균) 스무딩, alpha=0.2                           │
│    NormalizerProcessorStep                                            │
│       - TACTILE 은 NormalizationMode.IDENTITY (정규화 안 함)          │
│    Pi05PrepareStateTokenizerProcessorStep + tokenizer                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POLICY  src/lerobot/policies/pi05/modeling_pi05.py  (PI05Pytorch)    │
│    PaliGemma(vision+language) + Gemma action expert + flow matching   │
│                                                                       │
│    촉각 주입 모드 3가지 (config.tactile_injection_mode):              │
│      A0  "none"         → 촉각 미사용 (baseline)                      │
│      A15 "prefix"       → TactileTokenEncoder → prefix 토큰으로 삽입  │
│      A2  "action_gated" → ActionSideGatedTactileEncoder → expert gate │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                    PolicyAction (chunk_size=50 예측)
                                 │
                                 ▼
                        RobotAction (실행)
```

## 구성 요소

### 1. 촉각 하드웨어 추상화 — `src/lerobot_robot_paxini/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `paxini_so101.py` | `PaxiniSO101Follower` | SO-101 로봇 래퍼. 촉각 reader/renderer를 붙여 `observation.tactile.primary` 를 생성 |
| `paxini_reader.py` | `PaxiniReader`(base), `SerialPaxiniReader`, `CsvPaxiniReader`, `MockPaxiniReader` | 실 하드웨어(921600 baud) / CSV 리플레이 / 테스트용 mock. 154 taxel 파싱 |
| `tactile_render.py` | `TactileHeatmapRenderer`, `taxels_to_stacked_9x9()`, `TaxelMap`, `load_taxel_map_xlsx()` | taxel 좌표 → 64×64 heatmap 또는 (2,9,9) stacked grid 변환 |
| `config_paxini_so101.py` | `PaxiniSO101FollowerConfig` | `@RobotConfig.register_subclass("paxini_so101_follower")`. `tactile_port`, `tactile_num_taxels=154`, `tactile_representation`, `tactile_image_size=64`, `tactile_taxel_value_mode="z"` |
| `make_pi05_smoke_dataset.py` | — | smoke 테스트용 데이터셋 생성 |

**표현(representation) 두 가지**
- `"stacked_9x9"` → `(2, 9, 9)` : 두 센서(77 taxel)를 각각 9×9 그리드로, 채널 2로 stack
- `"heatmap"` → `(64, 64)` : taxel 좌표를 픽셀 그리드에 nearest-neighbor 보간
- 값 축(`tactile_taxel_value_mode`): `z`(수직력, 기본) / `x` / `y` / `magnitude`

### 2. 촉각 인코더 — `src/lerobot/policies/tactile/encoder.py`

policy 간 재사용 가능한 촉각 backbone. 입력 `(C,H,W)` → 토큰 임베딩.

| 클래스 | 설명 |
|--------|------|
| `TactileCNN` | 기본 CNN backbone, 샘플당 feature vector 1개 |
| `TactileAttentionCNN` | CNN + spatial attention |
| `DualPaxini3DCNN` | 실험용 3D(시간축) 인코더, 5D 입력 대응 |
| `TactileTokenEncoder` | 위 backbone을 감싸 `n_tokens` 개의 토큰으로 출력하는 래퍼 (prefix 모드에서 사용) |

### 3. pi0.5 policy 통합 — `src/lerobot/policies/pi05/`

| 파일 | 핵심 | 설명 |
|------|------|------|
| `configuration_pi05.py` | `PI05Config` | `@PreTrainedConfig.register_subclass("pi05")`. 촉각 필드: `tactile_encoder_type="cnn"`, `tactile_n_tokens=1`, `tactile_injection_mode`, `tactile_contact_threshold=0.05`, `tactile_dropout=0.3` |
| `modeling_pi05.py` | `PI05Pytorch`, `ActionSideGatedTactileEncoder` | PaliGemma + action expert. injection 모드에 따라 인코더 초기화(L641-664) |
| `processor_pi05.py` | `Pi05PrepareStateTokenizerProcessorStep`, `make_pi05_pre_post_processors()` | 전/후처리 파이프라인 구성 |

**촉각 주입 모드 상세**

- **A0 `none`**: 촉각 feature 미사용. `TACTILE` feature는 config에서 필터링.
- **A15 `prefix`**: `TactileTokenEncoder`가 `(2,9,9)`를 PaliGemma width 차원의 토큰으로 인코딩 → language prompt 임베딩 앞에 prefix로 삽입. `config.tactile_features` 각각에 대해 인코더를 `nn.ModuleList` 로 생성(L642-655).
- **A2 `action_gated`**: `ActionSideGatedTactileEncoder`(L553-609)가 9×9를 3×3 블록으로 나눠 블록별 5개 통계(mean/max/std/delta/active_ratio)와 전역 contact 통계를 뽑아 gating MLP로 action expert conditioning을 게이팅(L657-663).

### 4. 카메라 백엔드 — `src/lerobot/cameras/`

| 백엔드 | 파일 | 등록 이름 | 용도 |
|--------|------|-----------|------|
| AVFoundation | `avfoundation/camera_avfoundation.py`, `configuration_avfoundation.py` | `"avfoundation"` | macOS 네이티브 카메라(PyObjC) |
| WebRTC | `webrtc/camera_webrtc.py`, `configuration_webrtc.py` | `"webrtc"` | WHEP/WebRTC 저지연 원격 스트림(aiortc) |
| 기타 | `opencv/`, `realsense/`, `reachy2_camera/`, `zmq/` | — | 기존 LeRobot 백엔드 |

### 5. 추론/평가 스크립트

| 파일 | 역할 |
|------|------|
| `src/lerobot/async_inference/policy_server.py` | gRPC 비동기 policy 추론 서버. policy + 전/후처리 로드, observation 큐 관리 |
| `user_scripts/server.sh` | base pi05 / adapter / PaliGemma tokenizer 다운로드·경로 패치, `tactile_injection_mode` ↔ MODEL_TYPE(none/cnn/gated) 검증 후 gRPC 서버(0.0.0.0:8080) 기동 |
| `src/lerobot/scripts/lerobot_offline_action_eval.py` | 로봇 없이 녹화 에피소드에서 policy 예측 vs 시연 action 비교. chunk MAE/RMSE, 첫 action 오차, 방향 일치도, 관절별 motion jump 등 지표 계산 후 JSON/NPZ/plot 저장 |
| `user_scripts/offline_eval.sh` | `server.sh`(자산 준비) 후 A0/A15/A2 3개 체크포인트로 offline eval 실행. ENV: `EPISODE`, `STRIDE`, `MAX_FRAMES` |
| `experiments/pi05_cup_left/` | PEFT adapter 학습 하네스(A0/A15/A2 변형). prepare/preflight/run/queue 스크립트 |

## 설정 클래스 등록 관계

```
PreTrainedConfig
  └─ PI05Config                        register_subclass("pi05")

RobotConfig
  └─ PaxiniSO101FollowerConfig         register_subclass("paxini_so101_follower")

CameraConfig
  ├─ AVFoundationCameraConfig          register_subclass("avfoundation")
  └─ WebRTCCameraConfig                register_subclass("webrtc")

ProcessorStepRegistry
  ├─ "tactile_validation"              → TactileValidationProcessorStep
  ├─ "tactile_temporal_filter"         → TactileTemporalFilterProcessorStep
  └─ "pi05_prepare_state_tokenizer_processor_step" → Pi05PrepareStateTokenizerProcessorStep

FeatureType (src/lerobot/configs/types.py:27)
  └─ TACTILE = "TACTILE"
```

## 핵심 상수

- `OBS_TACTILE = "observation.tactile"` — `src/lerobot/utils/constants.py:29`
- `PAXINI_TACTILE_KEY = "observation.tactile.primary"` — `src/lerobot_robot_paxini/paxini_so101.py`
- 촉각 정규화: `NormalizationMode.IDENTITY` (min-max 스케일 안 함)
- pi0.5: `chunk_size=50`, `n_action_steps=50`, flow matching `num_inference_steps=10`

## 관련 문서

- `docs/tactile_pi05/README.md` — pi0.5 tactile rollout 최소 절차(한국어)
- `src/lerobot_robot_paxini/PAXINI_PI05_GUIDE.ko.md` — Paxini + pi0.5 가이드
- `experiments/pi05_cup_left/README.md` — A0/A15/A2 학습 하네스 가이드

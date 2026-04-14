# Autonomous-Driving-Research

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-0a7ea4.svg)](https://carla.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76b900.svg)](https://developer.nvidia.com/tensorrt)

Research platform for autonomous driving experiments in CARLA, combining:

- Interactive simulation runtime
- Replay-to-dataset generation pipeline
- Multi-model training playground
- Async closed-loop inference (PyTorch/TensorRT)

Current focus: dataset collection and curation from CARLA logs.

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Verify CLI

```bash
uv run main.py --help
```

### 3. Run a mode

Manual driving and optional recording:

```bash
uv run main.py --sync --delay 0.05 manual --record log
```

Replay and collect dataset samples:

```bash
uv run main.py --sync --delay 0.025 --fps 70 --timeout 30 \
  replay --replay-dir log/Town01/recording_YYYYMMDD_HHMMSS \
  --collect-data model/Autonomous_Dataset/carla/LAWM2 --headless
```

Inference in simulation:

```bash
uv run main.py --sync inference --model-path path/to/best_ModelName_runX.pt
```

## Project Scope

This repository supports the full loop:

1. Drive and record trajectories in CARLA.
2. Replay runs to align waypoints and export training data.
3. Train models from multiple architecture families.
4. Deploy models back into the simulation runtime for online evaluation.

At this stage, the repository is primarily used for steps 1 and 2 (data generation).

## Architecture

### High-Level Pipeline

```text
CARLA World -> Sensors -> Viewer/Control Loop -> Message Bus
      |                                     |
      |                                     +-> Inference (PyTorch/TensorRT)
      +-> Recorder + TrajectoryBuffer
                |
                v
          ReplayHandler -> Waypoint Alignment/Transforms -> Dataset Export
```

### Core Runtime Layers

- `main.py`: entrypoint, shared args, mode dispatch
- `mode/`: mode-specific orchestration (`manual`, `replay`, `inference`)
- `src/control/`: world wrapper, vehicle control, controller, sensor manager
- `src/spawn/`: spawning for actors and sensors
- `src/render/`: viewer loop, HUD, map overlays
- `src/math/`: projection, path optimization, transforms, branching logic
- `src/messages/`: typed pub/sub channels for telemetry and controls
- `src/others/`: replay processing, trajectory buffering, dataset writers
- `config/`: typed dataclass configuration

### Model and Training Layers

- `model/inference.py`: async model loading/inference orchestration
- `model/tensor_engine.py`: TensorRT engine utilities
- `model/*/train_script.py`: model-family training scripts
- `model/Autonomous_Dataset/`: dataset preparation/cleanup tooling

### Native Acceleration

- `fastmodel/` with `CMakeLists.txt` for LibTorch C++ experiments

## Modes

- `manual`: human driving, NPC spawning, optional recorder + trajectory logging
- `replay`: log playback, waypoint reconstruction, optional data collection
- `inference`: online model inference integrated in control/render loop

## Current Status

This repository currently does not publish model-performance metrics.

Primary use right now:

- collecting and replaying CARLA recordings
- exporting synchronized images + trajectory metadata
- preparing datasets for future training and evaluation

For external sharing, you can include qualitative media only (recording clips, route snapshots, and dataset examples) until benchmark results are available.

## Repository Layout

- `main.py` - top-level CLI entry
- `record.sh` - batch replay/data-collection helper
- `config/` - runtime and simulation constants
- `mode/` - mode runners
- `src/` - simulation runtime implementation
- `model/` - model training/inference ecosystem
- `fastmodel/` - native C++ experiments
- `log/` - local recordings (ignored by git)

## Configuration

Primary runtime config lives in `config/config.py` and includes:

- synchronization and fixed-step timing
- GPS delay/noise behavior
- replay stabilization and timing controls
- map rendering and waypoint offsets
- data collection cadence and thresholds

## Tech Stack

- Python 3.10
- CARLA 0.9.16
- PyTorch 2.x
- TensorRT 10.x
- OpenCV / NumPy / Rich / Pygame

## Development

Run checks:

```bash
uv run pytest
uv run ruff check .
```

Optional profiling output is written to `profile_results.lprof`.

## License

No top-level license file is currently present.
Add one before external distribution.

# Copilot instructions for LAWM

## Scope and working directory
- These instructions are for `LAWM/` only (inside a larger monorepo).
- Run commands from `LAWM/` so relative config/data paths in `cfgs/` and `csv_metadata/` resolve correctly.

## Big-picture architecture
- Entry point: `app/main.py` reads YAML (`--fname`) and dispatches to `app.{app}.train` (`app: action|probe|pretraining`).
- Training code is split by app into `app/<mode>/train.py` plus `app/<mode>/compile/{models,dataloader,optim,transform}.py`.
- Core model building blocks live in `models/`:
  - `vision_transformer.py` (encoder/tokenization)
  - `action_predictor.py` (goal-conditioned action latents)
  - `latent_dreamer.py` (action-conditioned latent rollout)
- Masked pretraining path uses `masks/multiseq_multiblock3d.py` (`MaskCollator`) and model wrappers in `models/utils/`.

## Data and metadata flow (important)
- CSVs in `csv_metadata/{action,probe,pretrain}/` map labels + sample directories (`datasets/utils/load_helper.py::_load_samples_and_labels`).
- Dataset classes in `datasets/dataset.py`:
  - `VideoDataset` → clips only
  - `ActVideoDataset` → context clips + prediction clips + GT metadata
  - `ProbeDataset` → full clips + GT metadata
- For action/probe, each sequence must contain images and `.npy` metadata; `_check_structure` searches recursively for both.
- `.npy` files are expected to contain image references and keys like `steer` and `velocity` (`datasets/utils/decode.py`, `_find_metadata_values`).

## Developer workflows used in this repo
- Main launcher (single or multi-GPU process spawn):
  - `python app/main.py --fname cfgs/action/action-224px-1024.24e.yaml --devices cuda:0`
  - `python app/main.py --fname cfgs/probe/probe-224px-1024.24e.yaml --devices cuda:0 cuda:1`
- Outputs/checkpoints go under `Experiment/<mode>/run*/weights` via `training_logger.py` + `early_stop.py`; config YAML is copied into the run folder.
- Quick dataset sanity check/visualization: `python datasets/dataset.py` (uses OpenCV frame stepping in `__main__`).

## Project-specific conventions to preserve
- Keep the compile pattern: wire new knobs through `cfgs/*.yaml` → `train.py` unpacking → `compile/*.py` function args.
- Preserve rank-0-only side effects (run dir creation, checkpoint saving, verbose logging) in distributed sections.
- `torch.hub.load(... facebookresearch/vjepa2 ...)` in action/probe compile is a hard integration point; avoid changing model tuple assumptions without checking all call sites.
- Do not “clean up” absolute imports blindly: some modules intentionally reference cross-root paths like `model.training_logger` / `model.early_stop` and legacy `model.JEPA_ACT.*` namespaces.

## Known sharp edges (verify before changing)
- Config key drift exists: `action/probe/train.py` reads `optim_cfg['annel']` and `loss_cfg['normalize_rep']`; example YAMLs may use `anneal` / `normalize_reps`.
- There are spelling variants like `persistance_workers` in compile signatures (called with `persistent_workers` from train files).
- `decode_batch` currently uses `ThreadPoolExecutor(max_workers=1)`; performance changes here affect all dataset variants.

## External dependencies/integration points
- Python deps are declared in repo-level `pyproject.toml` (outside `LAWM/`).
- Optional JPEG acceleration uses `TurboJPEG` with `/usr/lib/libturbojpeg.so.0`; falls back to OpenCV/PIL.
- Distributed setup uses `utils/distributed.py` (`nccl`, SLURM-aware fallback, rank/world-size detection).
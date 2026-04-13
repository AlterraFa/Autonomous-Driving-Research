Action Latent Waypoint Probe — LAWM Straightening Pipeline
==========================================================

Trains a lightweight **waypoint decoder** on top of a **fully frozen
Latent Action World Model** (LAWM).  The world model was previously
trained with the straightening objective (`app/straightening/`), which
jointly learns a video encoder filter, a goal-conditioned action predictor
(IDM), and an action-conditioned latent dreamer (FDM).  This probe stage
freezes the entire world model and trains *only* the decoder head to
predict **K future waypoints** in Frenet coordinates from the world
model's internal action latent representations.

The core hypothesis: if the action latents learned by LAWM encode
sufficient information about the agent's plan, a small decoder should
be able to recover the ego-vehicle's future trajectory — without ever
seeing raw pixels at decode time.


Architecture
------------

### Frozen World Model (`FrozenWorldModel`)

The frozen backbone is composed of four sub-networks loaded from a
straightening checkpoint:

| Component               | Class                             | Role                                         |
|-------------------------|-----------------------------------|----------------------------------------------|
| **Encoder**             | `VisionTransformer` (V-JEPA 2.1)  | Tokenize video frames → patch embeddings     |
| **Filterer**            | `TransformerStraightener`         | Temporal denoising / representation straightening |
| **Action Predictor**    | `ActionTransformerPredictorGC`    | Goal-conditioned IDM: (context, goal) → action latents |
| **Latent Predictor**    | `VisionTransformerPredictorAC`    | Action-conditioned FDM: (state, action) → next-state |

All parameters are frozen (`requires_grad=False`) and the model is kept
in `eval()` mode throughout probe training.

### Decoder (trainable)

Two decoder variants are supported:

**ActionDecoder** (Transformer Decoder)
- Projects action latent tokens into `d_model`-dim space via a linear layer
- `n_waypoints` learnable query tokens + sinusoidal positional encoding
- Standard `nn.TransformerDecoder` with `n_layers` layers of
  self-attention → cross-attention (queries attend to action memory)
- Final LayerNorm → 2-layer MLP head (d_model → mlp_hidden → 2)
- Output: `[B, n_waypoints, 2]` directly

**EfficientProbe** (Cross-Attention Pooler)
- Adapted from the VJEPA probing architecture (`models/probes/efficient.py`)
- Treats action tokens as a 1D temporal sequence (`num_patches=1`,
  `tubelet_size=1`)
- `num_queries` learnable query tokens per timestep, with temporal
  interpolation (`interp_q`) to match the actual action step count
- `UnitEncoding` positional embeddings (NeRF-style coordinate projection)
- `EfficientAttention` cross-attention pools queries against action tokens
- Optional self-attention `Block` layers for deeper processing (`depth > 1`)
- `output_dim` independent linear heads (embed → embed/2 → 1), concatenated
- Output: `[B, n_waypoints * 2]` → reshaped to `[B, n_waypoints, 2]`


Pipeline (single training step)
-------------------------------

```
                               ┌───────────────────────────────────────────────┐
                               │           FROZEN WORLD MODEL                  │
    video [B,C,T,H,W]         │                                               │
         │                     │                                               │
         ├── frames[:-1] ──►  V-JEPA 2.1 Encoder  ──►  z_ctx                  │
         │                           (frozen)             │                    │
         └── frame[-1]  ──►  V-JEPA 2.1 Encoder  ──►  z_goal                  │
                                     (frozen)             │                    │
                                                          ▼                    │
                                               TransformerStraightener         │
                                              (temporal filter, frozen)        │
                                                     │          │              │
                                                  h_ctx      h_goal            │
                                                     │          │              │
                                                     ▼          ▼              │
                                              ActionTransformerPredictorGC     │
                                               (goal-conditioned IDM)          │
                                                          │                    │
                                                       a_tf                    │
                                               [B, T_act, 128]                │
                                                          │                    │
                                               ┌──────────┘                    │
                                               │  Autoregressive Rollout       │
                                               │  (repeat for auto_steps):     │
                                               │                               │
                                               │   z_ctx ──► LPred(z, a)       │
                                               │               │               │
                                               │            z_next              │
                                               │               │               │
                                               │   [z_ctx; z_next] ──►         │
                                               │         APred(z, goal) ──► a  │
                                               │                               │
                                               └───────────────────────────────┘
                                                          │
                                                   a_latent (final)
                                               [B, T_act, 128]
                                                          │
                  ┌───────────────────────────────────────┘
                  │           TRAINABLE DECODER
                  ▼
    ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
    │       ActionDecoder             │ OR │        EfficientProbe            │
    │                                 │    │                                  │
    │  la_proj(a) → memory            │    │  UnitEncoding + EfficientAttn   │
    │  queries + sinusoidal PE        │    │  query tokens attend to a       │
    │  TransformerDecoder (self+cross)│    │  optional Block layers (depth)  │
    │  LayerNorm → MLP → (x, y)      │    │  output_dim linear heads        │
    │                                 │    │  reshape → [B, K, 2]           │
    │  → [B, K, 2]                    │    │                                  │
    └─────────────────────────────────┘    └──────────────────────────────────┘
                  │
               pred_wp [B, K, 2]
                  │
                  ▼
    ┌──────────────────────────────────────────────────┐
    │                  FDATLoss                         │
    │                                                  │
    │   (pred_wp, gt_wp, gate_score)  →  L_total       │
    │                                                  │
    │   L_total = L_frenet                             │
    │           + λ_heading · L_heading                │
    │           + λ_smooth  · L_smooth                 │
    └──────────────────────────────────────────────────┘
                  │
             loss.backward()   ← gradients flow ONLY to decoder
```


Frozen Forward Pass — Detail
-----------------------------

Given a video clip `[B, C, T, H, W]` with `T` frames:

1. **Encode**:  Split into context frames `clips[:, :, :-1]` and goal
   frame `clips[:, :, -1:]`.  Each is tokenized by the V-JEPA 2.1
   ViT-Base encoder into patch embeddings of shape `[B, N_tokens, 768]`,
   where `N_tokens = (T_sub / tubelet_size) × (H / patch_size)²`.

2. **Straighten**:  The `TransformerStraightener` filters both context
   and goal embeddings.  This is a multi-head self-attention module
   (embed_dim → filter_dim → embed_dim with skip connections) that was
   trained to produce temporally smooth, prediction-friendly
   representations.  Optional LayerNorm is applied if `normalize_reps`
   is enabled.

3. **Action prediction (teacher-forced)**:  The
   `ActionTransformerPredictorGC` (Goal-Conditioned) takes filtered
   context `h_ctx` and goal `h_goal` and predicts action latents
   `a_tf ∈ ℝ^{B × T_act × 128}`.  It uses GCBlock layers with
   goal-conditioned cross-attention — the action latent at each step
   represents "what action bridges context towards goal."

4. **Autoregressive rollout** (for `auto_steps` iterations):
   - `VisionTransformerPredictorAC` (Action-Conditioned, i.e. the FDM)
     predicts next-frame latents: `z_next = LPred(z_ctx, a_tf)`.
   - The predicted `z_next` is appended to the context window.
   - A new action sequence is predicted from the extended context:
     `a_new = APred(z_extended, h_goal, T)`.
   - The final action latent tensor from the last rollout step is
     returned.

   This multi-step rollout means the action latents encode not just a
   single-step action but the full imagined plan to reach the goal frame.


Loss: Frenet-Decomposed Anisotropic Trajectory (FDAT)
-----------------------------------------------------

`FDATLoss` decomposes prediction error in the **Frenet frame** of the
ground-truth trajectory — splitting into along-track (tangential, `e_s`)
and cross-track (normal, `e_d`) components.

### Frenet decomposition

For each waypoint, the GT tangent vector `T` is estimated via central
differences.  The normal `N = [-T_y, T_x]`.  Error `e = pred - gt` is
projected:

$$e_s = e \cdot T \qquad \text{(along-track)}$$

$$e_d = e \cdot N \qquad \text{(cross-track)}$$

### Gate-conditioned dual mode

A binary `gate_score` (derived from `road_type`: `uni` → 0, `multi` → 1)
switches between two operating modes:

| Mode                   | Gate | Cross-track (`α`) | Along-track (`β`) | Endpoint |
|------------------------|------|--------------------|-------------------|----------|
| **Lane-following**     | 0.0  | `α_lane = 20.0`    | `β_lane = 1.0`    | —        |
| **Intersection**       | 1.0  | `α_inter = 10.0`   | `β_inter = 3.0`   | `λ_ep`   |

Lane-following mode penalizes cross-track error 20× more than
along-track (staying *in* the lane matters more than exact longitudinal
position).  Intersection mode relaxes this ratio and adds an endpoint
anchor loss to ensure the trajectory converges to the correct exit.

### Positional weighting (bathtub curve)

Waypoints near the start and end of the trajectory are weighted more
heavily:

$$w(i) = 1 + \exp(-i / \tau_{\text{start}}) + \exp(-(K-1-i) / \tau_{\text{end}})$$

Start waypoints are safety-critical (immediate actions); end waypoints
anchor the goal.

### Component losses

All errors use `SmoothL1Loss` (Huber) with configurable `β`:

$$L_{\text{frenet}} = (1-g) \cdot L_{\text{lane}} + g \cdot L_{\text{inter}}$$

$$L_{\text{heading}} = \text{mean}\big(1 - \cos(\Delta\theta_{\text{pred}}, \Delta\theta_{\text{gt}})\big)$$

$$L_{\text{smooth}} = \text{mean}\big(\|\ddot{p}_{\text{pred}}\|^2\big) \qquad \text{(jerk penalty)}$$

$$L_{\text{total}} = L_{\text{frenet}} + \lambda_h \cdot L_{\text{heading}} + \lambda_s \cdot L_{\text{smooth}}$$


Data Pipeline
-------------

### StraighteningProbeDataset

Each dataset directory contains numbered `.npy` sequence files.  Each
`.npy` stores a dict with:

- `img_file`: ordered dict mapping frame indices to image paths
- `metadata.gt_data`: ground-truth signals (waypoints, velocity, etc.)
- `metadata.condition`: context tags (`road_type`: `'uni'` or `'multi'`)

At load time:

1. Images are decoded (JPEG → numpy) via `decode_batch` using
   `TurboJPEG` (falls back to OpenCV/PIL).
2. The shared transform is applied (resize, crop, normalize to ImageNet
   stats).
3. Waypoints are extracted from `metadata.gt_data[waypoint_key]`.  If
   the number of points ≠ `n_waypoints`, they are linearly interpolated
   to the target count.
4. `gate_score` is derived from `road_type` (`multi` → 1.0, else 0.0).

The dataset is split into train/val/test subsets (default 90/10/0) and
served via `DistributedSampler`-backed `DataLoader`s.


Optimization
------------

| Hyperparameter     | Default           | Notes                                    |
|--------------------|-------------------|------------------------------------------|
| Optimizer          | AdamW             | Separate param groups: bias/1D tensors get `wd=0` |
| LR scheduler       | CosineWSDSchedule | Warmup → Sustain → Cosine Decay          |
| WD scheduler       | CosineWDSchedule  | Cosine interpolation ref_wd → final_wd   |
| Mixed precision    | bfloat16          | `torch.amp.autocast` + `GradScaler`      |
| Gradient clipping  | None              | —                                        |

### CosineWSDSchedule (Warmup–Sustain–Decay)

```
  LR
   │
ref ├───────────────────────────────╮
   │      ╱                         ╲
   │     ╱  warmup        sustain    ╲  cosine decay
   │    ╱                             ╲
   │   ╱                               ╲
start ╱                                 ╲───── final_lr
   │
   └──────────────────────────────────────── step
        warmup_steps     T_max      anneal_steps
```

**Warmup**: linear `start_lr → ref_lr` over `warmup × ipe` steps.
**Sustain**: constant `ref_lr` for the middle portion.
**Decay**: cosine annealing `ref_lr → final_lr` over `anneal × ipe` steps.


Checkpointing & Early Stopping
-------------------------------

`EarlyStopping` tracks validation loss (mode `min`).  At each epoch end:

- If val loss improves by at least `min_delta`, save `best_decoder.pt`
  and reset the patience counter.
- Always save `last_decoder.pt` at the configured `save_every_freq`.
- A `checkpoint.pt` meta-file stores epoch, score, optimizer state, and
  scaler state for resumption.
- If patience is exhausted, `early_stop` flag is set (not currently used
  to break training, but available).

The run directory structure:

```
Experiment/probe/runN/
├── probe-action-{crop_size}px.yaml   # frozen copy of the full config
└── weights/
    ├── best_decoder.pt               # best val loss checkpoint
    ├── last_decoder.pt               # most recent checkpoint
    └── checkpoint.pt                 # meta (epoch, optimizer, scaler)
```


Training Loop Summary
---------------------

```
for epoch in range(epochs):

    # --- Train ---
    decoder.train()
    for step in range(ipe):
        clips, gt = next(dataloader)       # [B,C,T,H,W], {midlane_wp, gate_score}

        with autocast(bfloat16):
            with no_grad():
                a_latent = world_model(clips)   # [B, T_act, 128]

            pred_wp = decoder(a_latent)         # [B, K, 2]

            loss_dict = fdat_loss(pred_wp, gt_wp, gate_score)
            loss = loss_dict['total'].mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        lr_scheduler.step()
        wd_scheduler.step()

    # --- Validate ---
    decoder.eval()
    for clips, gt in val_loader:
        val_loss = val_step(clips, gt)

    # --- Checkpoint ---
    early_stopping(val_loss, decoder, epoch=epoch, optimizer=...)
```


Configs
-------

Two YAML configs are provided:

| Config                                 | Decoder          | Trainable params |
|----------------------------------------|------------------|------------------|
| `probe-action-latent-256px.yaml`       | ActionDecoder    | ~562K            |
| `probe-action-efficient-256px.yaml`    | EfficientProbe   | ~456K            |

Both auto-load the encoder + straightening architecture from the source
run's saved YAML (via `_load_run_config`).  Only the decoder section and
training hyperparameters need to be specified.


Running
-------

### Single-GPU

```bash
python -m app.main --fname cfgs/probe/probe-action-latent-256px.yaml --devices cuda:0
```

```bash
python -m app.main --fname cfgs/probe/probe-action-efficient-256px.yaml --devices cuda:0
```

### Multi-GPU (DDP)

```bash
python -m app.main --fname cfgs/probe/probe-action-latent-256px.yaml --devices cuda:0 cuda:1
```

The decoder is wrapped in `DistributedDataParallel`; the frozen world
model is replicated without DDP since it has no gradients.

### Resume from checkpoint

Add `continue_from_path` to the `meta` section of the YAML:

```yaml
meta:
  continue_from_path: ./Experiment/probe/run5
  resume_prefer_best: true   # load best_decoder.pt (false → last_decoder.pt)
```


Logging
-------

A Rich rolling table is printed every batch during training with columns:

| LR | WD | Total | Frenet | Heading | Smooth |
|----|-----|-------|--------|---------|--------|

At epoch end, validation metrics are appended.  If `save_csv: true`,
per-epoch CSVs are written to the run directory.


File Map
--------

```
app/probe/
├── train.py                          # Main training loop
├── compile/
│   ├── __init__.py                   # Re-exports compile functions
│   ├── models.py                     # FrozenWorldModel + ActionDecoder + EfficientProbe wiring
│   ├── dataloader.py                 # StraighteningProbeDataset → DataLoaders
│   ├── loss.py                       # FDATLoss + compile_fdat_loss
│   ├── optim.py                      # AdamW + CosineWSDSchedule + CosineWDSchedule
│   └── transform.py                  # Video augmentation / normalization transforms
├── eval.py                           # (evaluation utilities)
└── PIPELINE.md                       # This file

cfgs/probe/
├── probe-action-latent-256px.yaml    # ActionDecoder variant
└── probe-action-efficient-256px.yaml # EfficientProbe variant

datasets/dataset.py                   # StraighteningProbeDataset class
models/
├── action_predictor.py               # ActionTransformerPredictorGC (IDM)
├── latent_dreamer.py                 # VisionTransformerPredictorAC (FDM)
├── straightening_filter.py           # TransformerStraightener / MambaStraightener
├── vision_transformer.py             # V-JEPA 2.1 ViT encoder
└── probes/efficient.py               # EfficientProbe (cross-attention pooler)
```

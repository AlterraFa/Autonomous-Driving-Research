import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import inspect
import os
from ruamel.yaml import YAML
from models.vision_transformer import VisionTransformer as Enc
from models.latent_dreamer import VisionTransformerPredictorAC as LPred
from models.action_predictor import ActionTransformerPredictorGC as APred
from models.straightening_filter import FILTER_REGISTRY, Aggregation
from models.probes import available_probe_classes, build_probe, get_probe_class

from utils.logger import Logger
from torch import distributed as dist

logger = Logger(__name__)


def _load_run_config(run_dir: str) -> dict:
    """Discover and load the YAML config saved inside a training run directory.

    Searches for ``*.yaml`` / ``*.yml`` files directly inside *run_dir*
    (not recursively).  If multiple YAML files exist, the first one
    alphabetically is used.

    Returns the parsed dict (empty dict if nothing found).
    """
    candidates = sorted(
        glob.glob(os.path.join(run_dir, "*.yaml"))
        + glob.glob(os.path.join(run_dir, "*.yml"))
    )
    if not candidates:
        logger.WARNING(f"No YAML config found in {run_dir}")
        return {}
    path = candidates[0]
    logger.INFO(f"Auto-loading run config from {path}")
    yaml = YAML(typ="safe")
    with open(path) as f:
        return yaml.load(f) or {}


def sinusoidal_positional_encoding(n_positions: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(n_positions, d_model)
    position = torch.arange(0, n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ActionDecoder(nn.Module):
    """Transformer Decoder that maps latent action z -> n_waypoints waypoints (x, y).

    z (latent action) serves as Memory/Context for cross-attention.
    Learnable queries + sinusoidal PE -> self-attn + cross-attn -> MLP -> (x, y)
    """

    def __init__(
        self,
        la_dim: int,
        n_waypoints: int = 12,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.la_proj = nn.Linear(la_dim, d_model)
        self.queries = nn.Parameter(torch.zeros(1, n_waypoints, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.register_buffer(
            "pos_enc",
            sinusoidal_positional_encoding(n_waypoints, d_model).unsqueeze(0),
            persistent=False,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, la: torch.Tensor) -> torch.Tensor:
        """
        Args:
            la: [B, T_act, la_dim] or [B, la_dim] latent action tokens
        Returns:
            waypoints: [B, n_waypoints, 2]
        """
        if la.ndim == 2:
            la = la.unsqueeze(1)
        B = la.shape[0]
        memory = self.la_proj(la)
        q = self.queries.expand(B, -1, -1) + self.pos_enc
        q = self.transformer(tgt=q, memory=memory)
        q = self.norm(q)
        return self.mlp_head(q)

class FrozenWorldModel(nn.Module):
    """Wraps the frozen straightening world model components for inference only.
    
    Runs: clips -> encoder -> filterer -> apred (action predictor) -> action latents
    With optional autoregressive rollout via lpred.
    """

    def __init__(
        self,
        encoder: nn.Module,
        filterer: nn.Module,
        target_filterer: nn.Module,
        apred: nn.Module,
        lpred: nn.Module,
        patch_size: int = 16,
        tokens_pframe: int = 256,
        auto_steps: int = 5,
        normalize_reps: bool = True,
        normalize_actions: bool = False,
        detailed_out: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.filterer = filterer
        self.target_filterer = target_filterer
        self.apred = apred
        self.lpred = lpred
        self.patch_size = patch_size
        self.tokens_pframe = tokens_pframe
        self.auto_steps = auto_steps
        self.normalize_reps = normalize_reps
        self.normalize_actions = normalize_actions
        self.detailed_out = detailed_out

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, context: torch.Tensor, goal: torch.Tensor, perturb_coeff = 0.0):
        """Fully autoregressive forward from first frame (context) + last frame (goal).

        Only encodes frame 0 and frame -1. Rolls out action + prediction
        autoregressively for auto_steps-1 iterations.

        Args:
            clips: [B, C, T, H, W] full video clip
        Returns:
            detailed_out=False → a_tf  [B, T_act, action_embed_dim]
            detailed_out=True  → (a_tf, z_ar)
        """
        B, C, T_frames, H, W = context.shape

        # --- Encode only first and last frame ---
        latent_ctx  = self.encoder(context)
        latent_goal = self.encoder(goal)
        if self.normalize_reps:
            latent_ctx  = F.layer_norm(latent_ctx,  (latent_ctx.size(-1),))
            latent_goal = F.layer_norm(latent_goal, (latent_goal.size(-1),))

        # --- Filter: online for context, EMA target for goal ---
        h_ctx  = self.filterer(latent_ctx)
        h_goal = self.target_filterer(latent_goal)
        if self.normalize_reps:
            h_ctx  = F.layer_norm(h_ctx,  (h_ctx.size(-1),))
            h_goal = F.layer_norm(h_goal, (h_goal.size(-1),))

        # --- Fully autoregressive rollout from frame 0 ---
        z_ctx = h_ctx                                   # single frame tokens
        for _ in range(self.auto_steps - 1):
            a = self.apred(z_ctx, h_goal)
            
            # if perturb_coeff > 0.0:
            #     if not hasattr(self, "apstep"):
            #         self.apstep = a.shape[1]
            #     noise = torch.randn(a[:, -self.apstep:].shape, device = a.device) * perturb_coeff
            #     a[:, -self.apstep:] += noise
                
            
            if self.normalize_actions:
                a = F.layer_norm(a, (a.size(-1),))
            z_nxt = self.lpred(z_ctx, a)
            if self.normalize_reps:
                z_nxt = F.layer_norm(z_nxt, (z_nxt.size(-1),))
            z_nxt = z_nxt[:, -self.tokens_pframe:]
            z_ctx = torch.cat([z_ctx, z_nxt], dim=1)
            a_tf = a

        if self.detailed_out:
            return a_tf, z_ctx
        return a_tf


def _load_straightening_ckpt(checkpoint_dir, models_dict, map_location, prefer_best=True):
    prefix = "best_" if prefer_best else "last_"
    for name, model in models_dict.items():
        model_path = os.path.join(checkpoint_dir, f"{prefix}{name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        state = torch.load(model_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state)
    return models_dict


def compile_model(
    enc_cfg: dict = None,
    probe_cfg: dict = None,
    world_model_cfg: dict = None,
    device=torch.device('cpu'),
    detailed_out = False
):
    """Compile world model (frozen) + waypoint decoder (trainable).

    If ``world_model_cfg['checkpoint_dir']`` points to a training run that
    contains a saved YAML config, the encoder / filter / pred / action
    sections are read from that YAML automatically.  Any keys explicitly
    provided in *enc_cfg* or *world_model_cfg* override the auto-loaded
    values.

    Returns:
        world_model: FrozenWorldModel instance (all frozen)
        decoder: ActionDecoder instance (trainable)
    """
    enc_cfg = dict(enc_cfg or {})
    probe_cfg = dict(probe_cfg or {})
    world_model_cfg = dict(world_model_cfg or {})

    # ---- Auto-load config from the source run directory ----------------
    ckpt_dir = world_model_cfg.get('checkpoint_dir')
    if ckpt_dir:
        run_dir = ckpt_dir
        if os.path.basename(run_dir) == "weights":
            run_dir = os.path.dirname(run_dir)
        run_cfg = _load_run_config(run_dir)
        if run_cfg:
            run_model = run_cfg.get('model', {})
            run_loss = run_cfg.get('loss', {})
            run_common = run_cfg.get('common', run_model.get('common', {}))

            # Encoder: run config as base, explicit enc_cfg overrides
            auto_enc = dict(run_model.get('enc', {}))
            auto_enc.update({k: v for k, v in enc_cfg.items() if v is not None})
            enc_cfg = auto_enc

            # World-model sub-configs: only fill in missing keys
            for section in ('filter', 'pred', 'action', 'common'):
                if section not in world_model_cfg or not world_model_cfg[section]:
                    world_model_cfg[section] = run_model.get(section, {})

            # Loss-derived defaults (normalize_reps, auto_steps, etc.)
            if 'normalize_reps' not in world_model_cfg:
                world_model_cfg['normalize_reps'] = run_loss.get(
                    'normalize_reps', run_loss.get('normalize_rep', True)
                )
            if 'normalize_actions' not in world_model_cfg:
                world_model_cfg['normalize_actions'] = run_loss.get('normalize_actions', False)
            if 'auto_steps' not in world_model_cfg:
                world_model_cfg['auto_steps'] = run_loss.get('auto_steps', 6)

            logger.INFO(f"Merged run config from {run_dir} into compile_model args")
    world_model_cfg = world_model_cfg or {}

    # -- Load encoder
    name = enc_cfg.get('name', "Not found")
    repo = enc_cfg.get('load_from', 'Not found')
    source = enc_cfg.get('source', 'github')

    logger.INFO(f"Loading encoder from {source}")
    model = torch.hub.load(repo, name, source=source, pretrained=False, trust_repo=True, skip_validation=True)
    encoder: Enc = model[0]
    encoder.use_activation_checkpointing = enc_cfg.get('use_activation_checkpointing', True)

    if hasattr(encoder, 'img_size'):
        encoder.img_size = enc_cfg.get('crop_size', 224)
    if hasattr(encoder, 'patch_size'):
        encoder.patch_size = enc_cfg.get('patch_size', 16)
    if hasattr(encoder, 'tubelet_size'):
        encoder.tubelet_size = enc_cfg.get('tubelet_size', 2)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)

    # -- Build straightening components from world_model_cfg
    filter_cfg = dict(world_model_cfg.get('filter', {}))
    lpred_cfg = dict(world_model_cfg.get('pred', {}))
    apred_cfg = dict(world_model_cfg.get('action', {}))
    common_cfg = world_model_cfg.get('common', {})

    patch_size = enc_cfg.get('patch_size', 16)
    crop_size = enc_cfg.get('crop_size', 256)
    tubelet_size = enc_cfg.get('tubelet_size', 2)
    fpcs = enc_cfg.get('fpcs', 12)
    tokens_pframe = (crop_size // patch_size) ** 2

    filter_cfg['embed_dim'] = encoder.embed_dim
    filter_cfg['img_size'] = filter_cfg['crop_size']
    filterer = FILTER_REGISTRY[filter_cfg.get('name')](**filter_cfg).to(device)
    target_filterer = copy.deepcopy(filterer).to(device)

    # Latent predictor
    latent_predictor = LPred(
        img_size=lpred_cfg.get('crop_size', crop_size),
        patch_size=lpred_cfg.get('patch_size', patch_size),
        num_frames=lpred_cfg.get('fpcs', fpcs),
        tubelet_size=lpred_cfg.get('tubelet_size', tubelet_size),
        action_pframe=common_cfg.get('action_pframe', 1),
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=lpred_cfg.get('pred_embed_dim', 512),
        depth=lpred_cfg.get('depth', 6),
        num_heads=lpred_cfg.get('num_heads', 8),
        mlp_ratio=lpred_cfg.get('mlp_ratio', 4.0),
        qkv_bias=lpred_cfg.get('qkv_bias', True),
        qk_scale=lpred_cfg.get('qk_scale', None),
        drop_rate=lpred_cfg.get('drop_rate', 0.0),
        attn_drop_rate=lpred_cfg.get('attn_drop_rate', 0.0),
        drop_path_rate=lpred_cfg.get('drop_path_rate', 0.0),
        norm_layer=lpred_cfg.get('norm_layer', 'LayerNorm'),
        init_std=lpred_cfg.get('init_std', 0.1),
        uniform_power=lpred_cfg.get('uniform_power', True),
        use_silu=lpred_cfg.get('use_silu', False),
        wide_silu=lpred_cfg.get('wide_silu', True),
        is_frame_causal=lpred_cfg.get('is_frame_causal', True),
        use_activation_checkpointing=lpred_cfg.get('use_activation_checkpointing', False),
        use_rope=lpred_cfg.get('use_rope', True),
        action_embed_dim=apred_cfg.get('action_embed_dim', 128),
        use_sdpa=lpred_cfg.get('use_sdpa', False),
        out_norm=lpred_cfg.get('out_norm', 'LayerNorm'),
    ).to(device)

    # Action predictor
    action_predictor = APred(
        img_size=apred_cfg.get('crop_size', crop_size),
        patch_size=apred_cfg.get('patch_size', patch_size),
        max_frames=apred_cfg.get('fpcs', fpcs),
        tubelet_size=apred_cfg.get('tubelet_size', tubelet_size),
        action_per_step=common_cfg.get('action_pframe', 1),
        embed_dim=encoder.embed_dim,
        action_embed_dim=apred_cfg.get('action_embed_dim', 128),
        depth=apred_cfg.get('depth', 6),
        num_heads=apred_cfg.get('num_heads', 8),
        mlp_ratio=apred_cfg.get('mlp_ratio', 4.0),
        qkv_bias=apred_cfg.get('qkv_bias', True),
        qk_scale=apred_cfg.get('qk_scale', None),
        drop_rate=apred_cfg.get('drop_rate', 0.0),
        attn_drop_rate=apred_cfg.get('attn_drop_rate', 0.0),
        drop_path_rate=apred_cfg.get('drop_path_rate', 0.0),
        norm_layer=apred_cfg.get('norm_layer', 'LayerNorm'),
        init_std=apred_cfg.get('init_std', 0.1),
        uniform_power=apred_cfg.get('uniform_power', True),
        use_silu=apred_cfg.get('use_silu', False),
        wide_silu=apred_cfg.get('wide_silu', True),
        use_activation_checkpointing=apred_cfg.get('use_activation_checkpointing', False),
        use_rope=apred_cfg.get('use_rope', True),
        use_sdpa=apred_cfg.get('use_sdpa', False),
        out_norm=apred_cfg.get('out_norm', 'LayerNorm'),
    ).to(device)

    # Load checkpoint
    ckpt_dir = world_model_cfg.get('checkpoint_dir')
    if not ckpt_dir:
        raise ValueError("world_model.checkpoint_dir must be set to load straightening weights")
    weights_dir = os.path.join(ckpt_dir, "weights") if not ckpt_dir.endswith("weights") else ckpt_dir

    _load_straightening_ckpt(
        weights_dir,
        {
            "filter": filterer,
            "target_filter": target_filterer,
            "lpred": latent_predictor,
            "apred": action_predictor,
        },
        map_location=device,
        prefer_best=world_model_cfg.get('prefer_best', True),
    )
    logger.INFO(f"Loaded straightening checkpoint from {weights_dir}")

    normalize_reps = world_model_cfg.get('normalize_reps', True)
    normalize_actions = world_model_cfg.get('normalize_actions', False)
    auto_steps = world_model_cfg.get('auto_steps', fpcs // tubelet_size)

    world_model = FrozenWorldModel(
        encoder=encoder,
        filterer=filterer,
        target_filterer=target_filterer,
        apred=action_predictor,
        lpred=latent_predictor,
        patch_size=patch_size,
        tokens_pframe=tokens_pframe,
        auto_steps=auto_steps,
        normalize_reps=normalize_reps,
        normalize_actions=normalize_actions,
        detailed_out=detailed_out
    ).to(device)

    # -- Build waypoint decoder
    action_embed_dim = apred_cfg.get('action_embed_dim', 128)
    decoder_cfg = probe_cfg.get('decoder', {})
    decoder_type = decoder_cfg.get('type', 'ActionDecoder')

    if decoder_type == 'ActionDecoder':
        decoder = ActionDecoder(
            la_dim=action_embed_dim,
            n_waypoints=decoder_cfg.get('n_waypoints', 12),
            d_model=decoder_cfg.get('d_model', 256),
            n_heads=decoder_cfg.get('n_heads', 8),
            n_layers=decoder_cfg.get('n_layers', 4),
            dim_feedforward=decoder_cfg.get('dim_feedforward', 512),
            mlp_hidden=decoder_cfg.get('mlp_hidden', 128),
            dropout=decoder_cfg.get('dropout', 0.1),
        ).to(device)
    elif decoder_type == 'EfficientProbe':
        # Use EfficientProbe: treats action tokens like patch tokens
        decoder = build_probe(
            name='EfficientProbe',
            output_dim=decoder_cfg.get('n_waypoints', 12) * 2,
            embed_dim=action_embed_dim,
            num_patches=1,  # action tokens don't have spatial patches
            max_frames=fpcs,
            tubelet_size=1,  # each action token = 1 step
            num_heads=decoder_cfg.get('n_heads', 8),
            num_queries=decoder_cfg.get('num_queries', 16),
            depth=decoder_cfg.get('depth', 2),
            mlp_ratio=decoder_cfg.get('mlp_ratio', 4.0),
        ).to(device)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.INFO(f"World model parameters (frozen): {sum(p.numel() for p in world_model.parameters())}")
    logger.INFO(f"Decoder parameters (trainable): {count_params(decoder)}")

    return world_model, decoder


def compile_decoder_only(
    probe_cfg: dict = None,
    world_model_cfg: dict = None,
    device=torch.device('cpu'),
):
    """Build only the waypoint decoder (no world model). For cached-latent training.

    Reads `action_embed_dim` from the saved run config to match the cached
    latent dimension, then builds the decoder.

    Returns:
        decoder: ActionDecoder instance (trainable)
    """
    probe_cfg = dict(probe_cfg or {})
    world_model_cfg = dict(world_model_cfg or {})

    # Auto-load action_embed_dim from the source run config
    ckpt_dir = world_model_cfg.get('checkpoint_dir')
    action_embed_dim = 128  # default
    if ckpt_dir:
        run_dir = ckpt_dir
        if os.path.basename(run_dir) == "weights":
            run_dir = os.path.dirname(run_dir)
        run_cfg = _load_run_config(run_dir)
        if run_cfg:
            run_model = run_cfg.get('model', {})
            apred_cfg = run_model.get('action', {})
            action_embed_dim = apred_cfg.get('action_embed_dim', 128)

    decoder_cfg = probe_cfg.get('decoder', {})
    decoder_type = decoder_cfg.get('type', 'ActionDecoder')

    if decoder_type == 'ActionDecoder':
        decoder = ActionDecoder(
            la_dim=action_embed_dim,
            n_waypoints=decoder_cfg.get('n_waypoints', 12),
            d_model=decoder_cfg.get('d_model', 256),
            n_heads=decoder_cfg.get('n_heads', 8),
            n_layers=decoder_cfg.get('n_layers', 4),
            dim_feedforward=decoder_cfg.get('dim_feedforward', 512),
            mlp_hidden=decoder_cfg.get('mlp_hidden', 128),
            dropout=decoder_cfg.get('dropout', 0.1),
        ).to(device)
    elif decoder_type == 'EfficientProbe':
        enc_cfg = {}
        if ckpt_dir:
            run_dir = ckpt_dir
            if os.path.basename(run_dir) == "weights":
                run_dir = os.path.dirname(run_dir)
            run_cfg = _load_run_config(run_dir)
            if run_cfg:
                enc_cfg = run_cfg.get('model', {}).get('enc', {})
        fpcs = enc_cfg.get('fpcs', 12)
        decoder = build_probe(
            name='EfficientProbe',
            output_dim=decoder_cfg.get('n_waypoints', 12) * 2,
            embed_dim=action_embed_dim,
            num_patches=1,
            max_frames=fpcs,
            tubelet_size=1,
            num_heads=decoder_cfg.get('n_heads', 8),
            num_queries=decoder_cfg.get('num_queries', 16),
            depth=decoder_cfg.get('depth', 2),
            mlp_ratio=decoder_cfg.get('mlp_ratio', 4.0),
        ).to(device)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.INFO(f"Decoder-only mode (cached latents)")
    logger.INFO(f"Decoder parameters (trainable): {count_params(decoder)}")
    logger.INFO(f"Action embed dim (from world model config): {action_embed_dim}")

    return decoder

import glob
import os
import argparse
import re
import cv2
import numpy as np
import torch
import yaml
import torch.nn.functional as F

from math import ceil
from augmenter.transforms_builder import VideoTransform
from rich import print
from app.straightening.compile import compile_model
from traceback import print_exc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dirs",
    type=str,
    required=True,
    nargs="+",
    help="Sequence dirs or parent dirs containing multiple sequences",
)
parser.add_argument(
    "--mode",
    type=str,
    default="all",
    choices=["restructure", "infer", "all"],
    help="Run metadata merge, latent inference, or both",
)
parser.add_argument(
    "--num-images",
    type=int,
    default=1,
    help="Amount of images for inference (-1 for loading all images from folder)",
)
parser.add_argument(
    "--seq-batch-size",
    type=int,
    default=1,
    help="Number of sequences to process per batch",
)


def build_data_file(data_dir):
    meta_dirs = sorted(
        glob.glob(os.path.join(data_dir, "metadata/*")),
        key=lambda x: x.split("/")[-1],
    )

    if len(meta_dirs) == 0:
        raise FileNotFoundError(f"No metadata files found in {os.path.join(data_dir, 'metadata')}")

    merged_metadata = {
        "image_dir": [],
        "action": [],
        "pose": [],
        "timestamp": [],
    }

    for meta_dir in meta_dirs:
        metadata = np.load(meta_dir, allow_pickle=True).item()
        gt_data = metadata["metadata"]["gt_data"]

        steer = gt_data["steer"]
        throttle = gt_data["throttle"]
        brake = gt_data["brake"]

        condition = metadata["metadata"]["condition"]
        xy = np.asarray(condition["GPS"][:2])
        heading = condition["heading"]

        time_stamp = metadata["metadata"]["timestamp"]
        image_dir = metadata["img_file"]["I0"]

        merged_metadata["image_dir"].append(image_dir)
        merged_metadata["pose"].append(np.concatenate([xy, np.array([heading])]))
        merged_metadata["action"].append(np.array([steer, throttle, brake]))
        merged_metadata["timestamp"].append(time_stamp)

    merged_metadata["pose"] = np.array(merged_metadata["pose"])
    merged_metadata["action"] = np.array(merged_metadata["action"])
    merged_metadata["timestamp"] = np.array(merged_metadata["timestamp"])

    output_path = os.path.join(data_dir, "data.npy")
    np.save(output_path, merged_metadata, allow_pickle=True)
    print(f"Built merged metadata: {output_path}")


def _run_sort_key(run_path: str):
    run_name = os.path.basename(os.path.normpath(run_path))
    match = re.search(r"(\d+)$", run_name)
    if match is not None:
        return int(match.group(1))
    return run_name


def list_run_dirs(root_dir: str):
    run_dirs = [
        d for d in glob.glob(os.path.join(root_dir, "run*"))
        if os.path.isdir(d)
    ]

    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No run directories found in {root_dir}")

    return sorted(run_dirs, key=_run_sort_key)


def _is_sequence_dir(path: str):
    metadata_dir = os.path.join(path, "metadata")
    return os.path.isdir(metadata_dir)


def resolve_data_dirs(data_dirs):
    resolved = []
    visited = set()

    for raw_path in data_dirs:
        input_path = os.path.normpath(raw_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input data dir does not exist: {raw_path}")

        if _is_sequence_dir(input_path):
            if input_path not in visited:
                resolved.append(input_path)
                visited.add(input_path)
            continue

        found = []
        for root, dirs, _ in os.walk(input_path):
            if "metadata" in dirs:
                seq_dir = os.path.normpath(root)
                if seq_dir not in visited:
                    found.append(seq_dir)
                    visited.add(seq_dir)

        if len(found) == 0:
            raise FileNotFoundError(
                f"No sequence dirs with metadata/ found under: {input_path}"
            )

        resolved.extend(sorted(found))

    if len(resolved) == 0:
        raise ValueError("No valid sequence directories were resolved from --data-dirs")

    return resolved


def iter_sequence_batches(data_dirs, batch_size):
    if batch_size <= 0:
        raise ValueError("--seq-batch-size must be > 0")

    for start_idx in range(0, len(data_dirs), batch_size):
        yield data_dirs[start_idx:start_idx + batch_size]


def get_run_idx(run_dir: str):
    run_name = os.path.basename(os.path.normpath(run_dir))
    match = re.search(r"(\d+)$", run_name)
    if match is not None:
        return match.group(1)
    return run_name


def load_model(root_model, device):
    yaml_file = glob.glob(os.path.join(root_model, "*.yaml"))[0]

    with open(yaml_file, "r") as f:
        cfgs: dict = yaml.safe_load(f)

    model_cfgs = cfgs.get('model', {})
    enc_cfgs    = model_cfgs.get("enc", {})
    pred_cfgs   = model_cfgs.get('pred', {})
    act_cfgs    = model_cfgs.get('action', {})
    filter_cfgs = model_cfgs.get('filter', {})
    common_cfgs = model_cfgs.get('common', {})

    filter_weights = torch.load(os.path.join(root_model, "weights/best_target_filter.pt"))
    encoder, filterer, target_filterer, agg, *_ = compile_model(enc_cfg = enc_cfgs, lpred_cfg = pred_cfgs, apred_cfg = act_cfgs, filter_cfg = filter_cfgs, max_frames = 12, device = device)
    filterer.load_state_dict(filter_weights)
    agg.eval()
    encoder.eval()
    filterer.eval()
    return encoder, filterer, target_filterer, agg


def load_images(image_paths, amount):

    image_arr = []
    for idx, path in enumerate(image_paths):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        image_arr.append(image[None, ...])

        if amount != -1 and idx == amount - 1:
            break

    if len(image_arr) == 0:
        raise ValueError("No images were loaded for inference.")

    image_arr = np.concatenate(image_arr)

    return image_arr

def straighten_score(agg, h: torch.Tensor, tokens_pframe):
    B, _, D = h.shape
    _h = h.view(B, tokens_pframe, -1, D)
    _h = agg(_h)
    v = torch.diff(_h, dim = 2)
    v0 = v[:, :, :-1]
    v1 = v[:, :, 1:]
    cos_sim = torch.cosine_similarity(v0, v1, dim=-1)
    return (1 - cos_sim).mean()


def run_inference(data_dir, encoder, agg, filterer, transform, num_images, device, run_idx):
    metadata_path = os.path.join(data_dir, "data.npy")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing data file: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=True).item()
    image_paths = [os.path.join(data_dir, path) for path in metadata["image_dir"]]
    buffer = load_images(image_paths=image_paths, amount=num_images)
    num_loaded_images = buffer.shape[0]

    tubelet_size = encoder.tubelet_size
    embed_dim = encoder.embed_dim

    # remaining = num_loaded_images % tubelet_size
    # if remaining != 0:
    #     print(
    #         f"Number of images cannot be divided by tubelet size: {num_loaded_images} % {tubelet_size}.",
    #         f"Reducing to {num_loaded_images - remaining}",
    #     )
    #     buffer = buffer[:-remaining]
    #     num_loaded_images -= remaining

    if num_loaded_images == 0:
        raise ValueError(
            f"No usable frames remain after tubelet alignment for {data_dir} (tubelet_size={tubelet_size})."
        )

    T = ceil(num_loaded_images // tubelet_size)

    with torch.no_grad():
        with torch.autocast("cuda", dtype = torch.bfloat16):
            buffer = transform(buffer)[None, ...].to(device)
            latent = encoder(buffer)
            tokens_pframe = latent.numel() // (T * embed_dim)
            tokens_pdim = int(tokens_pframe ** 0.5)
            latent_straight = filterer(latent, tokens_pdim, tokens_pdim)

            be4_straight = straighten_score(agg, latent, tokens_pframe)            
            after_straight = straighten_score(agg, latent_straight, tokens_pframe)

            print(f"Run {run_idx} - Before: {be4_straight}, After: {after_straight}")

            if "z" not in metadata:
                metadata["z"] = latent.reshape(T, -1, embed_dim).to(torch.float16).cpu().numpy()
            metadata[f"z_straight{run_idx}"] = latent_straight.reshape(T, -1, embed_dim).to(torch.float16).cpu().numpy()
    np.save(metadata_path, metadata, allow_pickle=True)
    print(f"Done processing {data_dir} with run {run_idx}")


def _prepare_sequence(data_dir, num_images, tubelet_size):
    metadata_path = os.path.join(data_dir, "data.npy")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing data file: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=True).item()
    image_paths = [os.path.join(data_dir, path) for path in metadata["image_dir"]]
    buffer = load_images(image_paths=image_paths, amount=num_images)
    num_loaded_images = buffer.shape[0]

    # remaining = num_loaded_images % tubelet_size
    # if remaining != 0:
    #     buffer = buffer[:-remaining]
    #     num_loaded_images -= remaining

    if num_loaded_images == 0:
        raise ValueError(
            f"No usable frames remain after tubelet alignment for {data_dir} (tubelet_size={tubelet_size})."
        )

    return {
        "data_dir": data_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "buffer": buffer,
        "num_loaded_images": num_loaded_images,
    }

def forward_context(filterer, latent: torch.Tensor, H: int, W: int):
    h: torch.Tensor = filterer(latent, H, W)
    if True:
        h = F.layer_norm(h, (h.size(-1), ))
    return h

def forward_target(target_filterer, latent: torch.Tensor, H: int, W: int):
    with torch.no_grad():
        h: torch.Tensor = target_filterer(latent, H, W)
        if True:
            h = F.layer_norm(h, (h.size(-1), ))
        return h

def to_latent(encoder, c: torch.Tensor):
    with torch.no_grad():
        latent: torch.Tensor = encoder(c)
        if True:
            latent = F.layer_norm(latent, (latent.size(-1), ))
        return latent


def run_inference_batch(data_dirs, encoder, agg, filterer, target_filterer, transform, num_images, device, run_idx):
    tubelet_size = encoder.tubelet_size
    embed_dim = encoder.embed_dim

    seq_items = [_prepare_sequence(data_dir, num_images, tubelet_size) for data_dir in data_dirs]

    # Sequences in one tensor batch must share the same temporal length.
    common_num_images = min(item["num_loaded_images"] for item in seq_items)
    if common_num_images <= 0:
        raise ValueError("No valid frames available in sequence batch")

    trimmed_buffers = [item["buffer"][:common_num_images] for item in seq_items]
    T = ceil(common_num_images / tubelet_size)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batch_tensor = torch.stack([transform(buf) for buf in trimmed_buffers], dim=0).to(device)
            latent_ctx = to_latent(encoder, batch_tensor[:, :, :-1])
            latent_goal = to_latent(encoder, batch_tensor[:, :, -1:])
            latent = torch.cat([latent_ctx, latent_goal], dim = 1)
            
            
            B = latent.shape[0]
            tokens_pframe = latent.numel() // (B * T * embed_dim)
            tokens_pdim = int(tokens_pframe ** 0.5)
            
            context = forward_target(filterer, latent_ctx, tokens_pdim, tokens_pdim)
            goal = forward_target(target_filterer, latent_goal, tokens_pdim, tokens_pdim)
            latent_straight = torch.cat([context, goal], dim = 1)

            be4_straight = straighten_score(agg, latent_ctx, tokens_pframe)
            after_straight = straighten_score(agg, latent_straight, tokens_pframe)

            print(
                f"Run {run_idx} - Batch({len(seq_items)}) Before: {be4_straight}, "
                f"After: {after_straight}, T={T}"
            )

    for i, item in enumerate(seq_items):
        metadata = item["metadata"]
        metadata_path = item["metadata_path"]

        latent_i = latent[i]
        latent_straight_i = latent_straight[i]

        if "z" not in metadata:
            metadata["z"] = latent_i.reshape(T, -1, embed_dim).to(torch.float32).cpu().numpy()
        metadata[f"z_straight{run_idx}"] = latent_straight_i.reshape(T, -1, embed_dim).to(torch.float32).cpu().numpy()

        np.save(metadata_path, metadata, allow_pickle=True)
        print(f"Done processing {item['data_dir']} with run {run_idx}")


def main(args):
    data_dirs = resolve_data_dirs(args.data_dirs)
    print(f"Resolved {len(data_dirs)} sequence directories")
    print(f"Using sequence batch size: {args.seq_batch_size}")

    if args.mode in ["restructure", "all"]:
        for data_dir in data_dirs:
            build_data_file(data_dir)

    if args.mode in ["infer", "all"]:
        device = torch.device("cuda")
        run_dirs = list_run_dirs("./Experiment/straightening")

        transform = VideoTransform(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            crop_size=384,
            normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        for run_dir in run_dirs:
            try:
                run_idx = get_run_idx(run_dir)
                print(f"Loading model from {run_dir} (run_idx={run_idx})")
                encoder, filterer, target_filterer, agg = load_model(run_dir, device)

                total_batches = int(np.ceil(len(data_dirs) / args.seq_batch_size))
                for batch_idx, batch_data_dirs in enumerate(iter_sequence_batches(data_dirs, args.seq_batch_size), start=1):
                    print(f"Run {run_idx}: processing sequence batch {batch_idx}/{total_batches}")
                    run_inference_batch(
                        data_dirs=batch_data_dirs,
                        encoder=encoder,
                        agg=agg,
                        filterer=filterer,
                        target_filterer=target_filterer,
                        transform=transform,
                        num_images=args.num_images,
                        device=device,
                        run_idx=run_idx,
                    )
            except Exception as e:
                print_exc()


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
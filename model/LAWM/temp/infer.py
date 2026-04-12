import glob
import os
import argparse
import cv2
import numpy as np
import torch
import yaml

from augmenter.transforms_builder import VideoTransform
from rich import print
from app.straightening.compile import compile_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dirs",
    type=str,
    required=True,
    nargs="+",
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


def load_model(device):
    root_model = "./Experiment/straightening/run7"
    yaml_file = glob.glob(os.path.join(root_model, "*.yaml"))[0]

    with open(yaml_file, "r") as f:
        cfgs: dict = yaml.safe_load(f)

    model_cfgs = cfgs.get('model', {})
    enc_cfgs    = model_cfgs.get("enc", {})
    pred_cfgs   = model_cfgs.get('pred', {})
    act_cfgs    = model_cfgs.get('action', {})
    filter_cfgs = model_cfgs.get('filter', {})
    common_cfgs = model_cfgs.get('common', {})

    device = torch.device('cuda')
    filter_weights = torch.load(os.path.join(root_model, "weights/best_target_filter.pt"))
    encoder, filterer, _, agg, *_ = compile_model(enc_cfg = enc_cfgs, lpred_cfg = pred_cfgs, apred_cfg = act_cfgs, filter_cfg = filter_cfgs, max_frames = 12, device = device)
    filterer.load_state_dict(filter_weights)
    agg.eval()
    encoder.eval()
    filterer.eval()
    return encoder, filterer, agg


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


def run_inference(data_dir, encoder, agg, filterer, transform, num_images, device):
    metadata_path = os.path.join(data_dir, "data.npy")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing data file: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=True).item()
    image_paths = [os.path.join(data_dir, path) for path in metadata["image_dir"]]
    buffer = load_images(image_paths=image_paths, amount=num_images)
    num_loaded_images = buffer.shape[0]

    tubelet_size = encoder.tubelet_size
    embed_dim = encoder.embed_dim

    remaining = num_loaded_images % tubelet_size
    if remaining != 0:
        print(
            f"Number of images cannot be divided by tubelet size: {num_loaded_images} % {tubelet_size}.",
            f"Reducing to {num_loaded_images - remaining}",
        )
        buffer = buffer[:-remaining]
        num_loaded_images -= remaining

    if num_loaded_images == 0:
        raise ValueError(
            f"No usable frames remain after tubelet alignment for {data_dir} (tubelet_size={tubelet_size})."
        )

    T = num_loaded_images // tubelet_size

    with torch.no_grad():
        with torch.autocast("cuda", dtype = torch.bfloat16):
            buffer = transform(buffer)[None, ...].to(device)
            latent = encoder(buffer)
            tokens_pframe = latent.numel() // (T * embed_dim)
            tokens_pdim = int(tokens_pframe ** 0.5)
            latent_straight = filterer(latent, tokens_pdim, tokens_pdim)

            be4_straight = straighten_score(agg, latent, tokens_pframe)            
            after_straight = straighten_score(agg, latent_straight, tokens_pframe)

            print(f"Before: {be4_straight}, After: {after_straight}")

    metadata["z"] = latent.reshape(T, -1, embed_dim).to(torch.float32).cpu().numpy()
    metadata["z_straight"] = latent_straight.reshape(T, -1, embed_dim).to(torch.float32).cpu().numpy()
    np.save(metadata_path, metadata, allow_pickle=True)
    print(f"Done processing {data_dir}")


def main(args):
    data_dirs = args.data_dirs

    if args.mode in ["restructure", "all"]:
        for data_dir in data_dirs:
            build_data_file(data_dir)

    if args.mode in ["infer", "all"]:
        device = torch.device("cuda")
        encoder, filterer, agg = load_model(device)

        transform = VideoTransform(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            crop_size=384,
            normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        for data_dir in data_dirs:
            run_inference(
                data_dir=data_dir,
                encoder=encoder,
                agg=agg,
                filterer=filterer,                
                transform=transform,
                num_images=args.num_images,
                device=device
            )


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
import os
import argparse

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from cuml.manifold import UMAP as GPU_UMAP
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

TEST_REGISTRY = {}


def register_test(cls):
    TEST_REGISTRY[cls.__qualname__] = cls
    return cls


class BaseTest(ABC):
    n_components = 2
    umap_params = {
        "n_neighbors": 50,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42,
        "init": 'spectral',
    }

    def __init__(self, args):
        self.args = args

    def resolve_n_components(self):
        if self.args.n_components is not None:
            return self.args.n_components
        return self.n_components

    def make_umap(self, **overrides):
        params = dict(self.umap_params)
        params.update(overrides)
        params["n_components"] = self.resolve_n_components()
        return GPU_UMAP(**params)

    @staticmethod
    def load_latent(data_dir):
        npy_path = os.path.join(data_dir, "data.npy")
        data = np.load(npy_path, allow_pickle=True).item()
        latent = data["z"]

        if torch.is_tensor(latent):
            latent = latent.cpu()
        else:
            latent = torch.from_numpy(latent)

        latent = F.layer_norm(latent, (latent.shape[-1],)).numpy()
        return latent

    @staticmethod
    def load_latent_with_images(data_dir):
        npy_path = os.path.join(data_dir, "data.npy")
        data = np.load(npy_path, allow_pickle=True).item()
        latent = data["z"]
        image_dir = data.get("image_dir", None)

        if torch.is_tensor(latent):
            latent = latent.cpu()
        else:
            latent = torch.from_numpy(latent)

        latent = F.layer_norm(latent, (latent.shape[-1],)).numpy()

        resolved_image_paths = []
        if image_dir is not None:
            npy_root = os.path.dirname(npy_path)
            for p in list(image_dir):
                p_str = str(p)
                resolved_image_paths.append(p_str if os.path.isabs(p_str) else os.path.join(npy_root, p_str))

        return latent, resolved_image_paths

    @abstractmethod
    def run(self):
        pass


@register_test
class EventClustering(BaseTest):
    n_components = 2

    def run(self):
        latent = self.load_latent(self.args.data_dir)
        t_series = latent.mean(1)

        reducer = self.make_umap(
            n_neighbors=100,
            min_dist=0.05,
            random_state=45,
        )
        embedding = reducer.fit_transform(t_series)

        if embedding.shape[1] < 2:
            raise ValueError("test_1 requires n_components >= 2 for 2D plotting")

        plt.figure(figsize=(10, 7))
        sc = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=np.arange(len(embedding)),
            cmap="plasma",
            s=50,
            alpha=0.8,
        )

        plt.plot(embedding[:, 0], embedding[:, 1], color="black", alpha=0.2, linewidth=1)

        for i in range(0, len(embedding), 1):
            plt.annotate(
                str(i * 2),
                (embedding[i, 0], embedding[i, 1]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                fontweight="bold",
                alpha=0.8,
            )

        plt.colorbar(sc, label="Frame Index (Time)")
        plt.title("UMAP Projection of V-JEPA Latents")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.axis("equal")
        plt.show()


@register_test
class GlobalActivation(BaseTest):
    n_components = 2

    def run(self):
        print("Running embeddings analytics")
        latent = self.load_latent(self.args.data_dir)
        T, P, D = latent.shape
        latent = latent.reshape(-1, D)

        sample_labels = np.repeat(np.arange(T), P)
        patch_indices = np.tile(np.arange(P), T)

        reducer = self.make_umap(
            n_neighbors=100,
            min_dist=0.1,
            random_state=45,
        )
        embeddings = reducer.fit_transform(latent)

        if embeddings.shape[1] < 2:
            raise ValueError("test_2 requires n_components >= 2 for 2D plotting")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

        sc_time = axes[0].scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=sample_labels, cmap="viridis", s=1, alpha=0.15,
        )
        fig.colorbar(sc_time, ax=axes[0], label="Sample Index (Time)")
        axes[0].set_title("Token-wise UMAP colored by sample_labels")

        sc_patch = axes[1].scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=patch_indices, cmap="turbo", s=1, alpha=0.15,
        )
        fig.colorbar(sc_patch, ax=axes[1], label="Patch Index")
        axes[1].set_title("Token-wise UMAP colored by patch_indices")

        for ax in axes:
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()


@register_test
class VisionActivation(BaseTest):
    n_components = 3
    up_power = 16
    tubelet_size = 2

    def project_to_grid(self, latent: np.ndarray):
        T, P, D = latent.shape
        grid_size = int(np.sqrt(P))
        
        # 1. Reshape to flat patches for the reducer
        # DO NOT INTERPOLATE YET
        flat_latent = latent.reshape(-1, D)

        # 2. PCA Pre-reduction (Highly recommended for stability)
        # Reducing to 50 dims before UMAP removes noise that causes blur
        pca_reducer = PCA(n_components=min(50, D))
        flat_latent_reduced = pca_reducer.fit_transform(flat_latent)

        # 3. UMAP reduction on the raw patches
        reducer = self.make_umap(
            n_neighbors=50,
            min_dist=0.1,  # Increase slightly to prevent "tiny dots"
            metric='cosine'
        )
        projected = reducer.fit_transform(flat_latent_reduced)

        # 4. Scale to RGB
        scaler = MinMaxScaler()
        color_features = scaler.fit_transform(projected[:, :3])
        
        # 5. Reshape back to the original grid (e.g., 14x14)
        color_grid = color_features.reshape(T, grid_size, grid_size, 3)

        # 6. NOW INTERPOLATE the 3-channel color map
        # This preserves the sharp semantic boundaries found by UMAP
        interpolated_grid = self._interpolate_colors(color_grid, up_power=self.up_power)
        
        return interpolated_grid

    def _interpolate_colors(self, rgb_grid, up_power):
        """Interpolates the 3-channel RGB map, not the D-dim latent"""
        T, H, W, C = rgb_grid.shape
        # Move to torch for fast interpolation
        x = torch.from_numpy(rgb_grid).permute(0, 3, 1, 2).float()
        
        # Use 'nearest' if you want to see exact patch boundaries
        # Use 'bilinear' if you want a smooth heat-map look
        upsamp = F.interpolate(
            x,
            scale_factor=up_power,
            mode='bilinear', 
            align_corners=False # Set to False for patch-based data
        )
        return upsamp.permute(0, 2, 3, 1).numpy()

    def build_frame_image_pairs(self, image_paths, total_frames):
        required = total_frames * self.tubelet_size
        if len(image_paths) < required:
            raise ValueError(
                f"image_dir has {len(image_paths)} images, but need at least {required} "
                f"for {total_frames} latent frames with tubelet_size={self.tubelet_size}."
            )

        pairs = []
        for frame_idx in range(total_frames):
            start = frame_idx * self.tubelet_size
            pair = image_paths[start:start + self.tubelet_size]
            if len(pair) != self.tubelet_size:
                raise ValueError(f"Missing tubelet images for latent frame {frame_idx}")
            for p in pair:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Image path not found: {p}")
            pairs.append(pair)
        return pairs

    def interactive_frame_viewer(self, latent_grid, frame_image_pairs):
        total_frames = latent_grid.shape[0]
        frame_idx = int(np.clip(self.args.start_frame, 0, total_frames - 1))

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
        ax_img_1 = fig.add_subplot(gs[0, 0])
        ax_img_2 = fig.add_subplot(gs[0, 1])
        ax_latent = fig.add_subplot(gs[1, :])

        src_1 = mpimg.imread(frame_image_pairs[frame_idx][0])
        src_2 = mpimg.imread(frame_image_pairs[frame_idx][1])
        shown_src_1 = ax_img_1.imshow(src_1)
        shown_src_2 = ax_img_2.imshow(src_2)
        shown_latent = ax_latent.imshow(latent_grid[frame_idx], interpolation = 'nearest')

        ax_img_1.axis("off")
        ax_img_2.axis("off")
        ax_latent.axis("off")

        help_text = "Keys: ←/A prev | →/D next | ↑/W +10 | ↓/S -10 | Q quit"
        title = fig.suptitle(f"Frame {frame_idx + 1}/{total_frames}\n{help_text}")
        ax_img_1.set_title("Source image t")
        ax_img_2.set_title("Source image t+1")
        ax_latent.set_title("Latent UMAP RGB map")

        def update(new_idx):
            nonlocal frame_idx
            frame_idx = int(np.clip(new_idx, 0, total_frames - 1))
            src_1_update = mpimg.imread(frame_image_pairs[frame_idx][0])
            src_2_update = mpimg.imread(frame_image_pairs[frame_idx][1])
            shown_src_1.set_data(src_1_update)
            shown_src_2.set_data(src_2_update)
            shown_latent.set_data(latent_grid[frame_idx])
            title.set_text(f"Frame {frame_idx + 1}/{total_frames}\\n{help_text}")
            fig.canvas.draw_idle()

        def on_key(event):
            key = (event.key or "").lower()
            if key in ("left", "a"):
                update(frame_idx - 1)
            elif key in ("right", "d"):
                update(frame_idx + 1)
            elif key in ("up", "w"):
                update(frame_idx + 10)
            elif key in ("down", "s"):
                update(frame_idx - 10)
            elif key in ("q", "escape"):
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show()

    def run(self):
        latent, image_paths = self.load_latent_with_images(self.args.data_dir)
        latent_grid = self.project_to_grid(latent)
        frame_image_pairs = self.build_frame_image_pairs(image_paths, latent_grid.shape[0])

        self.interactive_frame_viewer(latent_grid, frame_image_pairs)

@register_test
class RegionalActivation(BaseTest):
    n_components = 2
    tubelet_size = 2

    def run(self):
        print("Running Interactive UMAP Slice Viewer")
        # 1. Load Data
        latent, image_paths = self.load_latent_with_images(self.args.data_dir)
        T, P, D = latent.shape

        # 2. Global UMAP Projection (Calculated once for consistency)
        print(f"Calculating global UMAP manifold for {T*P} tokens...")
        flat_latent = latent.reshape(-1, D)
        
        # PCA for stability
        pca_reducer = PCA(n_components=min(100, D))
        flat_reduced = pca_reducer.fit_transform(flat_latent)

        reducer = self.make_umap(n_neighbors=100, min_dist=0.1, spread = 3.0, metric='cosine')
        projected = reducer.fit_transform(flat_reduced)
        
        # Reshape to (T, P, 2) so we can slice by frame
        projected_frames = projected.reshape(T, P, 2)
        patch_indices = np.arange(P)

        # Or just re-use logic:
        pairs = []
        for i in range(T):
            pairs.append(image_paths[i * self.tubelet_size : (i + 1) * self.tubelet_size])

        # 4. Launch Viewer
        self.interactive_slice_viewer(projected_frames, pairs, patch_indices)

    def interactive_slice_viewer(self, projected_frames, pairs, patch_indices):
        total_frames = projected_frames.shape[0]
        frame_idx = 0
        P = len(patch_indices)
        grid_size = int(np.sqrt(P))

        # 1. Setup Layout: 2 rows, 3 columns
        # Top row: [Image t] [Image t+1] [Patch Map Legend]
        # Bottom row: [Global UMAP Plot (spanning all columns)]
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.8], width_ratios=[1, 1, 0.6])
        
        ax_img1 = fig.add_subplot(gs[0, 0])
        ax_img2 = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2]) # This replaces the colorbar
        ax_umap = fig.add_subplot(gs[1, :])

        # -- A. Setup Patch Map Legend (The "Image" of the colormap) --
        # We create a 2D grid of the indices to show where colors belong
        patch_grid = np.arange(P).reshape(grid_size, grid_size)
        ax_legend.imshow(patch_grid, cmap='turbo', interpolation='nearest')
        ax_legend.set_title("Patch Map Legend\n(Spatial Color Key)")
        ax_legend.axis('off')

        # -- B. Setup Background Shadow --
        all_x = projected_frames.reshape(-1, 2)[:, 0]
        all_y = projected_frames.reshape(-1, 2)[:, 1]
        ax_umap.scatter(all_x, all_y, c='lightgray', s=1, alpha=0.03, label='Global Manifold', zorder=1)

        # -- C. Setup Interactive Elements --
        src1 = mpimg.imread(pairs[frame_idx][0])
        src2 = mpimg.imread(pairs[frame_idx][1])
        show_img1 = ax_img1.imshow(src1)
        show_img2 = ax_img2.imshow(src2)
        
        # Plot current frame tokens
        current_x = projected_frames[frame_idx, :, 0]
        current_y = projected_frames[frame_idx, :, 1]
        show_scatter = ax_umap.scatter(current_x, current_y, c=patch_indices, 
                                      cmap='turbo', s=30, edgecolors='black', 
                                      linewidths=0.4, zorder=2)

        ax_img1.axis('off')
        ax_img2.axis('off')
        ax_img1.set_title("Source t")
        ax_img2.set_title("Source t+1")
        
        ax_umap.set_xlabel("UMAP Dimension 1")
        ax_umap.set_ylabel("UMAP Dimension 2")
        ax_umap.set_title("Current Frame Tokens on Global Manifold")
        
        title = fig.suptitle(f"Frame {frame_idx} | Keys: A/D to Navigate | Q to Quit", fontsize=14)

        def update(new_idx):
            nonlocal frame_idx
            frame_idx = int(np.clip(new_idx, 0, total_frames - 1))
            
            # Update Images
            show_img1.set_data(mpimg.imread(pairs[frame_idx][0]))
            show_img2.set_data(mpimg.imread(pairs[frame_idx][1]))
            
            # Update Scatter Positions
            new_pos = projected_frames[frame_idx]
            show_scatter.set_offsets(new_pos)
            
            title.set_text(f"Frame {frame_idx}/{total_frames-1} | Patch-wise UMAP Slice")
            fig.canvas.draw_idle()

        def on_key(event):
            key = (event.key or "").lower()
            if key in ('left', 'a'): update(frame_idx - 1)
            elif key in ('right', 'd'): update(frame_idx + 1)
            elif key in ('up', 'w'): update(frame_idx + 10)
            elif key in ('down', 's'): update(frame_idx - 10)
            elif key in ('q', 'escape'): plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="Data dir for latent analysis")
    parser.add_argument("--run-test", type=str, help="Uses which test to run", default="test_1")
    parser.add_argument("--list-test", action="store_true", help="List out all available test")
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=(
            "Override UMAP n_components. If omitted, each test uses its own default "
            "(e.g., test_1/test_2: 2, test_3: 3)."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index for Test3 interactive viewer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_test:
        num_test = len(TEST_REGISTRY)
        test_name = [name for name in TEST_REGISTRY.keys()]
        print(f"{num_test} are available: {', '.join(test_name)}")
        raise SystemExit(0)

    if args.data_dir is None:
        raise ValueError("You have not specified a data directory for analysis")

    if args.run_test not in TEST_REGISTRY:
        raise ValueError(
            f"Unknown test '{args.run_test}'. Available tests: {', '.join(TEST_REGISTRY.keys())}"
        )

    test_runner = TEST_REGISTRY[args.run_test](args)
    test_runner.run()
import time
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def colorized_by_z(xyz: np.ndarray) -> np.ndarray: 
    z = xyz[:, 2]
    zNorm = (z - z.min()) / (np.ptp(z) + 1e-6)
    return plt.get_cmap('viridis')(zNorm)[:, :3]

def update(vis: o3d.visualization.Visualizer, 
           pcd: o3d.geometry.PointCloud,
           delay: int):
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(delay)

def create_line_set(corners: np.ndarray, color=(1,0,0)) -> o3d.geometry.LineSet:
    """Build an Open3D LineSet from 8 corners (any coord frame)."""
    # 12 edges of a box
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    colors = [color for _ in edges]
    ls = o3d.geometry.LineSet(
        points  = o3d.utility.Vector3dVector(corners),
        lines   = o3d.utility.Vector2iVector(edges)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

class LIDARVisualizer:
    def __init__(self, range: int, window_size: tuple = (800, 600)):
        line = np.linspace(-range, range, 100)[:, None]
        horizontal_line = np.hstack([
            line,
            np.zeros_like(line),
            np.zeros_like(line)
        ])
        vertical_line = np.hstack([
            np.zeros_like(line),
            line, 
            np.zeros_like(line)
        ])
        init_cross = np.r_[vertical_line, horizontal_line]


        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(init_cross)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name = "LIDAR Visualizer",
            height = window_size[0], width = window_size[1],
            left = 50, top = 50
        )
        
        self.vis.add_geometry(self.pcd)
        self.vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05])
        self.vis.get_render_option().point_size = 1
        self.vis.get_render_option().show_coordinate_frame = True
        
        self.flip_y = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],  # Negate Y
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])
        
        
    def display(self, pcd: np.ndarray, intensity: np.ndarray = None, delay: float = 0):
        
        self.pcd.points = o3d.utility.Vector3dVector(pcd)
        if intensity is not None and len(intensity) != 0:
            self.pcd.colors = o3d.utility.Vector3dVector(self.intensity_to_heatmap(intensity))
        self.pcd.transform(self.flip_y)
        
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(delay)
        
        
    
    def intensity_to_heatmap(self, intensity_values: np.ndarray) -> np.ndarray:
        normalized = (intensity_values - np.min(intensity_values)) / (np.ptp(intensity_values) + 1e-6)
        cmap = plt.get_cmap("plasma")      # blue→green→yellow→red
        heatmap_colors = cmap(normalized)[:, :3]  # remove alpha
        return heatmap_colors
        
    def destroy_window(self):
        self.vis.destroy_window()
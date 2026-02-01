#!/usr/bin/env python3
"""
Route analysis visualization script.
Plots CARLA map with route paths, metrics, and model predictions.
"""

import os
import sys
import csv
import glob
import numpy as np
import cv2

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
from pathlib import Path

# Add parent to path
script_path = os.path.abspath(__file__)
folder = os.path.dirname(script_path)
parent = os.path.dirname(folder)
sys.path.insert(0, parent)

# Import CARLA modules conditionally
try:
    import carla
    from utils.control.world import World
    from utils.math.world_map import Map
    from utils.math.path import PathHandler
    from utils.render.hud import overlay_waypoints_on_map
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    
from utils.messages.logger import Logger


class RouteAnalyzer:
    """Analyze and visualize route execution metrics."""
    
    def __init__(self, map_name="Town01", headless=True, skip_carla=False):
        self.log = Logger()
        self.map_name = map_name
        self.headless = headless
        self.carla_available = False
        self.map_processor = None
        
        # Connect to CARLA (optional - can work without it)
        if not skip_carla and CARLA_AVAILABLE:
            try:
                self.client = carla.Client("localhost", 2000)
                self.client.set_timeout(10.0)
                self.world_obj = World(self.client, 8000)
                self.carla_map = self.world_obj.world.get_map()
                self.log.INFO(f"Connected to CARLA on map: {self.carla_map.name}")
                
                # Initialize map processor
                self.map_processor = Map(self.world_obj, rect_dim=(1, 1), range_=(100, 100), map_offset = (10, 10), invert_color = True)
                self.log.INFO("Map processor initialized")
                self.carla_available = True
            except Exception as e:
                self.log.WARNING(f"CARLA not available: {e}")
                self.log.WARNING("Will generate plots without CARLA map visualization")
        else:
            if not CARLA_AVAILABLE:
                self.log.INFO("CARLA SDK not available")
            else:
                self.log.INFO("CARLA connection skipped")
    
    def is_metrics_file(self, filepath):
        """Detect if a file contains metrics data rather than route path."""
        try:
            if filepath.endswith('.npy'):
                data = np.load(filepath, allow_pickle=True)
                # Metrics NPY: 2D array with ~5 columns (timestamp, frame, distance, deviation, completion)
                return data.ndim == 2 and data.shape[1] <= 6
            
            with open(filepath, 'r', errors='ignore') as f:
                first_line = f.readline().strip()
                # Metrics format: "distance, deviation, completion, [x y z]"
                return '[' in first_line and ',' in first_line
        except:
            return False
    
    def swap_if_needed(self, route_file, metrics_file):
        """Auto-swap files if they appear to be in the wrong order."""
        route_is_metrics = self.is_metrics_file(route_file) if route_file else False
        metrics_is_route = not self.is_metrics_file(metrics_file) if metrics_file else False
        
        if route_is_metrics and metrics_is_route:
            self.log.WARNING(f"Arguments appear to be swapped. Swapping {route_file} and {metrics_file}")
            return metrics_file, route_file
        
        return route_file, metrics_file
    
    def load_metrics(self, metrics_file):
        """Load metrics from CSV, NPY, or legacy text format (distance, deviation, completion, [x y z])."""
        metrics = []
        try:
            # Try NPY format first
            if metrics_file.endswith('.npy'):
                try:
                    data = np.load(metrics_file, allow_pickle=True)
                    self.log.INFO(f"Loaded NPY metrics file: {metrics_file}")
                    
                    # Handle different NPY formats
                    if data.ndim == 2:
                        # Array format: each row is [distance, deviation, completion, ...]
                        for idx, row in enumerate(data):
                            metrics.append({
                                'timestamp': idx,
                                'frame_id': idx,
                                'distance_travelled_m': float(row[0]) if len(row) > 0 else 0,
                                'deviation_m': float(row[1]) if len(row) > 1 else 0,
                                'route_completion_pct': float(row[2]) if len(row) > 2 else 0,
                                'position': np.array(row[3:6]) if len(row) > 5 else None
                            })
                    else:
                        self.log.WARNING(f"Unexpected NPY shape: {data.shape}")
                        return metrics
                    
                    self.log.INFO(f"Loaded {len(metrics)} metric entries from NPY file")
                    return metrics
                except Exception as e:
                    self.log.WARNING(f"Failed to load as NPY: {e}, trying text format...")
            
            # Try text/CSV format
            with open(metrics_file, 'r', errors='ignore') as f:
                first_line = f.readline().strip()
                f.seek(0)
                
                # Check if it's a CSV file (has header row)
                if 'timestamp' in first_line or 'distance' in first_line or 'frame' in first_line:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        metrics.append({
                            'timestamp': float(row.get('timestamp_unix', idx)),
                            'frame_id': int(row.get('frame_id', idx)),
                            'distance_travelled_m': float(row['distance_travelled_m']),
                            'deviation_m': float(row['deviation_m']),
                            'route_completion_pct': float(row['route_completion_pct']),
                            'position': None
                        })
                else:
                    # Legacy format: distance, deviation, completion, [x y z]
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # Parse: "distance, deviation, completion, [x y z]"
                            parts = line.split(',', 3)
                            distance = float(parts[0].strip())
                            deviation = float(parts[1].strip())
                            completion = float(parts[2].strip())
                            
                            # Parse position array if it exists
                            position = None
                            if len(parts) > 3:
                                pos_str = parts[3].strip()
                                # Remove brackets and split
                                pos_str = pos_str.strip('[]')
                                pos_vals = [float(x) for x in pos_str.split()]
                                position = np.array(pos_vals)
                            
                            metrics.append({
                                'timestamp': idx,
                                'frame_id': idx,
                                'distance_travelled_m': distance,
                                'deviation_m': deviation,
                                'route_completion_pct': completion,
                                'position': position
                            })
                        except ValueError:
                            continue
            
            self.log.INFO(f"Loaded {len(metrics)} metric entries from {metrics_file}")
            return metrics
        except Exception as e:
            self.log.ERROR(f"Failed to load metrics: {e}")
            return []
    
    def load_route_path(self, path_file):
        """Load route path from NPY or TXT file."""
        try:
            # Try NPY format first
            if path_file.endswith('.npy'):
                path = np.load(path_file, allow_pickle=True)
                self.log.INFO(f"Loaded route path with {len(path)} waypoints from {path_file}")
                return path
            
            # Try text format (space or comma separated) - but skip if it looks like metrics data
            with open(path_file, 'r') as f:
                first_line = f.readline().strip()
                # Check if this looks like metrics data (distance, deviation, completion format)
                if ',' in first_line:
                    parts = first_line.split(',')
                    if len(parts) >= 3 and '[' in first_line:
                        # Looks like metrics data: "0.0, 1.116..., 0.0, [x y z]"
                        self.log.WARNING(f"File {path_file} appears to be metrics data, not a route path")
                        return None
            
            # Parse as coordinate array
            path = np.loadtxt(path_file, delimiter=None)
            if path.ndim == 1:
                path = path.reshape(-1, 1)
            if path.shape[1] < 2:
                self.log.ERROR(f"Route path must have at least 2 columns (x, y), got {path.shape[1]}")
                return None
            self.log.INFO(f"Loaded route path with {len(path)} waypoints from {path_file}")
            return path
        except Exception as e:
            self.log.ERROR(f"Failed to load route path: {e}")
            return None
    
    def plot_route_on_map(self, route_path, metrics=None, save_path=None, width_ratio=0.5, spacing=0.1):
        """Plot route on CARLA map with metrics visualization.
        
        Args:
            width_ratio: Controls relative width of plots (0.0-1.0)
                        0.5 = equal width, <0.5 = map smaller, >0.5 = map larger
            spacing: Controls horizontal spacing between plots (0.0-1.0)
                    0.0 = no gap, 0.1 = small gap (default), 0.25 = larger gap
        """
        if route_path is None or len(route_path) == 0:
            self.log.ERROR("No valid route path provided")
            return
        
        # Calculate summary stats
        total_distance = 0
        final_completion = 0
        if metrics:
            distances = [m['distance_travelled_m'] for m in metrics]
            completions = [m['route_completion_pct'] for m in metrics]
            total_distance = distances[-1] if distances else 0
            final_completion = completions[-1] if completions else 0
        
        # Create figure - 1x2 layout with configurable width ratio
        # width_ratio: 0.5 = equal, <0.5 = map smaller, >0.5 = map larger
        map_width = width_ratio
        deviation_width = 1 - width_ratio
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[map_width, deviation_width], wspace=spacing)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        title = f'Route Analysis - {self.map_name}\n'
        title += f'Total Distance: {total_distance:.2f}m  |  Route Completion: {final_completion:.1f}%'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # ===== Subplot 1: Map with Route =====
        
        # Use CARLA map if available, otherwise just plot route
        if self.carla_available and self.map_processor is not None:
            try:
                map_image = self.map_processor.map_image.copy()
                map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
                H, W = map_image.shape[:2]
                
                # Scale and draw route to match map pixel coordinates
                route_scaled = route_path[:, :2].copy()
                route_scaled[:, 0] = route_scaled[:, 0] * self.map_processor.scale + self.map_processor.offset_x
                # For Y: CARLA world coords are negative, flip to positive for image coords
                # route_scaled[:, 1] = route_scaled[:, 1] * self.map_processor.scale + self.map_processor.offset_y - 1050
                route_scaled[:, 1] = -route_scaled[:, 1] * self.map_processor.scale + self.map_processor.offset_y
                route_scaled[:, 1] = (H + 20) - route_scaled[:, 1]
                
                # Debug: print coordinate ranges
                self.log.INFO(f"Map dimensions: W={W}, H={H}")
                self.log.INFO(f"Map processor scale={self.map_processor.scale}, offset_x={self.map_processor.offset_x}, offset_y={self.map_processor.offset_y}")
                self.log.INFO(f"Route X range (corrected): [{route_scaled[:, 0].min():.1f}, {route_scaled[:, 0].max():.1f}]")
                self.log.INFO(f"Route Y range (corrected): [{route_scaled[:, 1].min():.1f}, {route_scaled[:, 1].max():.1f}]")
                
                # Step 1: Display the map first
                ax1.imshow(map_image, origin='upper', aspect='auto', extent=[0, W, H, 0])
                
                # Step 2: Plot routes on top with explicit coordinates
                ax1.plot(route_scaled[:, 0], route_scaled[:, 1], 'r-', linewidth=4, 
                        label='Planned Route', alpha=1.0, zorder=10)
                ax1.scatter(route_scaled[0, 0], route_scaled[0, 1], c='green', s=200, 
                           marker='o', label='Start', zorder=15, edgecolors='white', linewidths=2)
                ax1.scatter(route_scaled[-1, 0], route_scaled[-1, 1], c='red', s=200, 
                           marker='^', label='End', zorder=15, edgecolors='white', linewidths=2)
                
            except Exception as e:
                self.log.ERROR(f"Failed to render CARLA map: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to simple scatter plot
                route_scaled = route_path[:, :2]
                ax1.scatter(route_scaled[:, 0], route_scaled[:, 1], alpha=0.3, s=1, c='gray')
        else:
            # Simple scatter plot without CARLA
            route_scaled = route_path[:, :2]
            ax1.scatter(route_scaled[:, 0], route_scaled[:, 1], alpha=0.3, s=1, c='gray', label='Map Points')
        
        # Plot actual trajectory if available
        if metrics and any(m['position'] is not None for m in metrics):
            positions = np.array([m['position'][:2] for m in metrics if m['position'] is not None])
            if len(positions) > 0:
                if self.carla_available and self.map_processor is not None:
                    traj_scaled = positions.copy()
                    traj_scaled[:, 0] = traj_scaled[:, 0] * self.map_processor.scale + self.map_processor.offset_x
                    # For Y: CARLA world coords are negative, flip to positive for image coords  
                    # traj_scaled[:, 1] = traj_scaled[:, 1] * self.map_processor.scale + self.map_processor.offset_y - 1050
                    traj_scaled[:, 1] = -traj_scaled[:, 1] * self.map_processor.scale + self.map_processor.offset_y
                    traj_scaled[:, 1] = (H + 20) - traj_scaled[:, 1]
                    
                    self.log.INFO(f"Trajectory X range (corrected): [{traj_scaled[:, 0].min():.1f}, {traj_scaled[:, 0].max():.1f}]")
                    self.log.INFO(f"Trajectory Y range (corrected): [{traj_scaled[:, 1].min():.1f}, {traj_scaled[:, 1].max():.1f}]")
                else:
                    traj_scaled = positions
                ax1.plot(traj_scaled[:, 0], traj_scaled[:, 1], 'b-', linewidth=3, 
                        alpha=1.0, label='Actual Trajectory', zorder=12)
                ax1.scatter(traj_scaled[0, 0], traj_scaled[0, 1], c='blue', s=150, 
                           marker='o', alpha=0.9, zorder=15, edgecolors='white', linewidths=2)
        
        ax1.set_title('Route on Global Map', fontsize=14, fontweight='bold')
        ax1.set_xlim(-50, W + 50 if self.carla_available and self.map_processor else None)
        ax1.set_ylim(H + 50 if self.carla_available and self.map_processor else None, -50)
        legend = ax1.legend(loc='upper right', fontsize=11, framealpha=0.8, 
                           facecolor='white', edgecolor='black', frameon=True)
        legend.set_zorder(20)  # Ensure legend is above all route elements
        
        # ===== Subplot 2: Deviation from Route =====
        if metrics:
            deviations = [m['deviation_m'] for m in metrics]
            frames = [m['frame_id'] for m in metrics]
            ax2.plot(frames, deviations, 'orange', linewidth=2, label='Lateral Deviation')
            ax2.fill_between(frames, 0, deviations, alpha=0.3, color='orange')
            ax2.set_xlabel('Frame ID', fontsize=12)
            ax2.set_ylabel('Deviation (m)', fontsize=12)
            ax2.set_title('Lateral Deviation from Route', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add stats
            max_dev = max(deviations) if deviations else 0
            avg_dev = np.mean(deviations) if deviations else 0
            ax2.text(0.02, 0.98, f'Max: {max_dev:.2f}m\nAvg: {avg_dev:.2f}m',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log.INFO(f"Saved plot to {save_path}")
        
        return fig, (ax1, ax2)
    
    def compare_routes(self, route_files, metrics_files=None, save_dir=None):
        """Compare multiple routes."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Route Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(route_files)))
        
        for idx, (route_file, color) in enumerate(zip(route_files, colors)):
            route = self.load_route_path(route_file)
            metrics = None
            
            if metrics_files and idx < len(metrics_files):
                metrics = self.load_metrics(metrics_files[idx])
            
            label = Path(route_file).stem
            
            # Plot routes
            if route is not None:
                route_scaled = route[:, :2] * self.map_processor.scale
                route_scaled[:, 0] += self.map_processor.offset_x
                route_scaled[:, 1] += self.map_processor.offset_y
                axes[0].plot(route_scaled[:, 0], route_scaled[:, 1], color=color, 
                            linewidth=2, label=label, alpha=0.7)
            
            # Plot metrics
            if metrics:
                completion = [m['route_completion_pct'] for m in metrics]
                frames = [m['frame_id'] for m in metrics]
                axes[1].plot(frames, completion, color=color, linewidth=2, label=label, alpha=0.7)
        
        # Format subplots
        axes[0].set_title('Route Paths')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Route Completion Progress')
        axes[1].set_xlabel('Frame ID')
        axes[1].set_ylabel('Completion (%)')
        axes[1].set_ylim([0, 110])
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'route_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log.INFO(f"Saved comparison plot to {save_path}")
        
        return fig, axes
    
    def export_metrics_summary(self, metrics, output_file):
        """Export summary statistics of metrics."""
        if not metrics:
            self.log.WARNING("No metrics to export")
            return
        
        distances = [m['distance_travelled_m'] for m in metrics]
        deviations = [m['deviation_m'] for m in metrics]
        completions = [m['route_completion_pct'] for m in metrics]
        
        summary = {
            'total_frames': len(metrics),
            'max_distance_m': max(distances),
            'avg_distance_rate_m_per_frame': np.mean(distances),
            'total_distance_m': distances[-1] if distances else 0,
            'max_deviation_m': max(deviations),
            'avg_deviation_m': np.mean(deviations),
            'min_deviation_m': min(deviations),
            'final_completion_pct': completions[-1] if completions else 0,
            'max_completion_pct': max(completions),
        }
        
        with open(output_file, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        self.log.INFO(f"Exported metrics summary to {output_file}")
        return summary


def main():
    parser = argparse.ArgumentParser(description='Route Analysis Visualization')
    parser.add_argument('--map', default='Town01', help='CARLA map name')
    parser.add_argument('--route', help='Path to route NPY or TXT file')
    parser.add_argument('--metrics', help='Path to metrics CSV, TXT, or NPY file')
    parser.add_argument('--analyze-dir', help='Directory containing analysis files')
    parser.add_argument('--output-dir', default='analysis_plots', help='Output directory for plots')
    parser.add_argument('--compare', action='store_true', help='Compare multiple routes')
    parser.add_argument('--skip-carla', action='store_true', help='Skip CARLA connection (for headless mode)')
    parser.add_argument('--show', action='store_true', help='Show plots in GUI (requires display)')
    parser.add_argument('--width-ratio', type=float, default=0.5, help='Width ratio for map vs deviation plot (0.0-1.0, default=0.5 for equal width)')
    parser.add_argument('--spacing', type=float, default=0.1, help='Horizontal spacing between plots (0.0-1.0, default=0.1, use 0.0 for minimum gap)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    try:
        analyzer = RouteAnalyzer(map_name=args.map, skip_carla=args.skip_carla)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        if not args.skip_carla:
            print("Try using --skip-carla for headless mode")
        return
    
    if args.analyze_dir:
        # Analyze directory with multiple runs
        print(f"Analyzing directory: {args.analyze_dir}")
        
        route_files = sorted(glob.glob(os.path.join(args.analyze_dir, '*path*.npy'))) + \
                     sorted(glob.glob(os.path.join(args.analyze_dir, '*path*.txt')))
        metrics_files = sorted(glob.glob(os.path.join(args.analyze_dir, '*.csv'))) + \
                       sorted(glob.glob(os.path.join(args.analyze_dir, '*.npy'))) + \
                       sorted(glob.glob(os.path.join(args.analyze_dir, '*.txt')))
        
        # Filter out path files from metrics
        metrics_files = [f for f in metrics_files if 'path' not in f.lower()]
        
        if args.compare and len(route_files) > 1:
            # Compare multiple routes
            analyzer.compare_routes(route_files, metrics_files, args.output_dir)
        else:
            # Analyze each route individually
            for route_file in route_files:
                route = analyzer.load_route_path(route_file)
                if route is None:
                    continue
                
                # Find corresponding metrics file by matching base names
                base_name = Path(route_file).stem.replace('_path', '')
                metrics = None
                for mf in metrics_files:
                    mf_base = Path(mf).stem
                    if base_name in mf_base or mf_base in base_name:
                        metrics = analyzer.load_metrics(mf)
                        if metrics:
                            break
                
                # Plot
                output_file = os.path.join(args.output_dir, f"{base_name}_analysis.png")
                result = analyzer.plot_route_on_map(route, metrics, output_file, args.width_ratio, args.spacing)
                if result:
                    plt.close(result[0] if isinstance(result, tuple) else result)
                
                # Export summary
                if metrics:
                    summary_file = os.path.join(args.output_dir, f"{base_name}_summary.txt")
                    analyzer.export_metrics_summary(metrics, summary_file)
    
    elif args.route and args.metrics:
        # Single route analysis
        # Auto-detect if files are swapped
        route_file, metrics_file = analyzer.swap_if_needed(args.route, args.metrics)
        
        route = analyzer.load_route_path(route_file)
        metrics = analyzer.load_metrics(metrics_file)
        
        if route is not None:
            output_file = os.path.join(args.output_dir, 'route_analysis.png')
            result = analyzer.plot_route_on_map(route, metrics, output_file, args.width_ratio, args.spacing)
            if result:
                plt.close(result[0] if isinstance(result, tuple) else result)
            
            if metrics:
                summary_file = os.path.join(args.output_dir, 'metrics_summary.txt')
                analyzer.export_metrics_summary(metrics, summary_file)
        else:
            print(f"Failed to load route from {route_file}")
            print(f"Try swapping the arguments or check file formats")
    
    else:
        # Use latest files from store directory
        store_dir = os.path.join(parent, 'store')
        route_files = sorted(glob.glob(os.path.join(store_dir, '*_path*.npy'))) + \
                     sorted(glob.glob(os.path.join(store_dir, '*_path*.txt')))
        metrics_files = sorted(glob.glob(os.path.join(store_dir, '*metrics*.csv'))) + \
                       sorted(glob.glob(os.path.join(store_dir, '*metrics*.npy')))
        
        if route_files and metrics_files:
            latest_route = route_files[-1]
            latest_metrics = metrics_files[-1]
            
            print(f"Using latest files:")
            print(f"  Route: {latest_route}")
            print(f"  Metrics: {latest_metrics}")
            
            route = analyzer.load_route_path(latest_route)
            metrics = analyzer.load_metrics(latest_metrics)
            
            if route is not None:
                output_file = os.path.join(args.output_dir, 'latest_route_analysis.png')
                result = analyzer.plot_route_on_map(route, metrics, output_file, args.width_ratio, args.spacing)
                if result:
                    plt.close(result[0] if isinstance(result, tuple) else result)
                
                if metrics:
                    summary_file = os.path.join(args.output_dir, 'latest_metrics_summary.txt')
                    analyzer.export_metrics_summary(metrics, summary_file)
        else:
            print("No route or metrics files found in store directory")
            print("Available commands:")
            print("  python template/plot_route_analysis.py --route <path.npy> --metrics <metrics.csv>")
            print("  python template/plot_route_analysis.py --analyze-dir <directory>")
            print("  python template/plot_route_analysis.py --help")
    
    # Only show plots if requested and display is available
    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plots: {e}")
            print(f"Plots have been saved to {args.output_dir}")
    else:
        print(f"\nPlots saved to {args.output_dir}")
        print("Use --show to display plots in GUI (requires X11/Wayland display)")


if __name__ == "__main__":
    main()

import numpy as np
from scipy.spatial.transform import Rotation as R


def rpy2ypr(rot_rpy_deg: np.ndarray) -> R:
    """Build rotation from CARLA/UE style [roll, pitch, yaw] degrees."""
    rpy = np.asarray(rot_rpy_deg, dtype=float)
    if rpy.ndim == 1:
        if rpy.shape[0] != 3:
            raise ValueError("Rotation must be shape (3,) as [roll, pitch, yaw]")
        roll, pitch, yaw = rpy
        return R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)

    if rpy.ndim == 2 and rpy.shape[1] == 3:
        ypr = rpy[:, [2, 1, 0]]
        return R.from_euler('ZYX', ypr, degrees=True)

    raise ValueError("Rotation must be shape (3,) or (N,3) in [roll, pitch, yaw] order")

def local_2_global(location: np.ndarray, points: np.ndarray, rotation: float):

    if len(location) == 3:
        x, y, z = location
    else:
        x, y = location
        z = 0.0
    c, s = np.cos(rotation), np.sin(rotation)

    T = np.array([
        [ c, -s, x],
        [ s,  c, -y],
        [ 0,  0, 1]
    ])

    pts = np.atleast_2d(points)
    has_z = pts.shape[1] >= 3

    pts_xy = np.hstack([pts[:, :2], np.ones((pts.shape[0], 1))])
    global_xy = (T @ pts_xy.T).T[:, :2]

    if has_z:
        global_pts = np.hstack([global_xy, (pts[:, 2:3] + z)])
        return global_pts if len(global_pts) > 1 else global_pts[0]

    return global_xy if len(global_xy) > 1 else global_xy[0]


def global_2_local(location: np.ndarray, point: np.ndarray, rot: float):
    x, y, z = location
    c, s = np.cos(rot), np.sin(rot)

    T = np.array([
        [ c,  s, -x*c - y*s],
        [ s, -c, -x*s + y*c],
        [ 0,  0,        1 ]
    ])

    pts = np.atleast_2d(point)
    has_z = pts.shape[1] >= 3

    pts_xy = np.hstack([pts[:, :2], np.ones((pts.shape[0], 1))])
    local_xy = (T @ pts_xy.T).T[:, :2]

    if has_z:
        local_pts = np.hstack([local_xy, (pts[:, 2:3] - z)])
        return local_pts if len(local_pts) > 1 else local_pts[0]

    return local_xy if len(local_xy) > 1 else local_xy[0]

def global_2_local_full_rot(location: np.ndarray, points: np.ndarray, rotation_deg: np.ndarray):
    """
    location: [x, y, z] of vehicle
    points: (N, 3) or (3,) global points
    rotation_deg: [roll, pitch, yaw] in degrees
    """
    loc = np.asarray(location)
    pts = np.atleast_2d(points)
    
    # 1. Translation
    rel_xyz = pts[:, :3] - loc
    
    # 2. Build the exact CARLA World-to-Local Matrix
    # We use radians for the trig functions
    r, p, y = np.radians(rotation_deg)
    
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    # This is the Transpose of the Local-to-World matrix.
    # It converts Global -> Local directly.
    world_to_local_mat = np.array([
        [cp * cy, cp * sy, sp],
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr],
        [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr]
    ])

    # 3. Apply Rotation (Matrix Multiplication)
    # rel_xyz is (N, 3), mat is (3, 3). Result is (N, 3)
    local_xyz = rel_xyz @ world_to_local_mat.T

    # Append any extra columns (like rotation) if they existed
    if pts.shape[1] > 3:
        return np.hstack([local_xyz, pts[:, 3:]])
    
    return local_xyz if len(local_xyz) > 1 else local_xyz[0]

def global_2_local_rot(global_rpy_deg, vehicle_rpy_deg):
    """
    Simple angular delta for Roll, Pitch, Yaw. 
    For turn prediction, we usually just need the Yaw delta.
    """
    # Normalize degrees to [-180, 180] to prevent 360-degree jumps
    diff = np.asarray(global_rpy_deg) - np.asarray(vehicle_rpy_deg)
    return (diff + 180) % 360 - 180
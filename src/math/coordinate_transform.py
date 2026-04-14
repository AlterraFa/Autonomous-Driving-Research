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


# ── Camera projection ──────────────────────────────────────────────────────────

def camera_intrinsic(width: int, height: int, fov_deg: float = 90.0) -> np.ndarray:
    """
    Build a 3×3 pinhole intrinsic matrix K from CARLA camera attributes.

    Parameters
    ----------
    width, height : int
        Image resolution — matches sensor.set_attribute('image_size_x/y', ...).
    fov_deg : float
        Horizontal field of view in degrees — matches sensor.set_attribute('fov', ...).
        CARLA default is 90.0.

    Returns
    -------
    K : np.ndarray, shape (3, 3)
    """
    f  = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    cx = width  / 2.0
    cy = height / 2.0
    return np.array([
        [f,   0.,  cx],
        [0.,  f,   cy],
        [0.,  0.,   1.],
    ], dtype=np.float64)


def camera_extrinsic(cam_transform: dict) -> np.ndarray:
    """
    Build a 4×4 extrinsic matrix that maps points from the vehicle ego local
    frame to the CARLA camera local frame (UE axes: X=fwd, Y=right, Z=up).

    Parameters
    ----------
    cam_transform : dict
        Camera pose relative to the vehicle, with keys:
        'x', 'y', 'z' (metres) and 'roll', 'pitch', 'yaw' (degrees).
        Matches the CameraView enum format used across the codebase, e.g.:
            CameraView.FIRST_PERSON.value  ->  {x:0, y:0, z:2, roll:0, pitch:0, yaw:0}

    Returns
    -------
    E : np.ndarray, shape (4, 4)
        Transforms homogeneous ego-frame points into camera UE-frame points.
    """
    x     = cam_transform.get("x",     0.0)
    y     = cam_transform.get("y",     0.0)
    z     = cam_transform.get("z",     0.0)
    roll  = cam_transform.get("roll",  0.0)
    pitch = cam_transform.get("pitch", 0.0)
    yaw   = cam_transform.get("yaw",   0.0)

    # Rotation of the camera expressed in ego frame (ZYX = yaw → pitch → roll)
    r_cam_in_ego = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    # Inverse rotation: ego frame → camera frame
    R_mat = r_cam_in_ego.inv().as_matrix()           # (3, 3)
    t_cam = np.array([x, y, z], dtype=np.float64)

    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_mat
    E[:3,  3] = -R_mat @ t_cam                       # translation in camera frame
    return E


# Remaps CARLA/UE axes (X=fwd, Y=right, Z=up) to standard CV axes (X=right, Y=down, Z=fwd).
# Applied before intrinsic projection so the standard formula u = f*X/Z + cx works correctly.
_UE_TO_CV = np.array([
    [0.,  1.,  0.],
    [0.,  0., -1.],
    [1.,  0.,  0.],
], dtype=np.float64)


def ego_to_pixel(
    points_ego: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    width: int,
    height: int,
    clip: bool = True,
) -> np.ndarray:
    """
    Project ego-frame 3-D waypoints to image pixel coordinates.

    Waypoints are expected in vehicle body frame as produced by
    ``global_2_local_full_rot`` (X=fwd, Y=right, Z=up). If only XY is
    available (as stored in the dataset), Z=0 is assumed.

    Parameters
    ----------
    points_ego : np.ndarray, shape (N, 3) or (N, 2)
        Points in vehicle ego local frame.
    K : np.ndarray, shape (3, 3)
        Intrinsic matrix from ``camera_intrinsic()``.
    E : np.ndarray, shape (4, 4)
        Extrinsic matrix from ``camera_extrinsic()``.
    width, height : int
        Image resolution used for optional boundary clipping.
    clip : bool
        When True, pixels that fall outside the image boundary are returned as
        np.nan rows.  Points behind the camera are always np.nan regardless.

    Returns
    -------
    pixels : np.ndarray, shape (N, 2), dtype float64
        [u, v] pairs — (column, row) in pixel space.
        Rows with np.nan mark invalid projections.

    Example
    -------
    >>> K = camera_intrinsic(1280, 720, fov_deg=90)
    >>> E = camera_extrinsic(CameraView.FIRST_PERSON.value)
    >>> uv = ego_to_pixel(local_loc_spatial, K, E, 1280, 720)
    """
    pts = np.atleast_2d(np.asarray(points_ego, dtype=np.float64))
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])

    # Homogeneous ego coords → camera UE frame
    ones     = np.ones((len(pts), 1), dtype=np.float64)
    p_cam_ue = (E @ np.hstack([pts[:, :3], ones]).T).T[:, :3]   # (N, 3)

    # UE axes → standard CV axes (X=right, Y=down, Z=fwd)
    p_cv = p_cam_ue @ _UE_TO_CV.T                               # (N, 3)

    # Only points with positive depth are in front of the camera
    valid    = p_cv[:, 2] > 0.0
    pixels   = np.full((len(pts), 2), np.nan, dtype=np.float64)

    if valid.any():
        p_v  = p_cv[valid]
        uv   = (K @ (p_v / p_v[:, 2:3]).T).T[:, :2]            # (M, 2)

        if clip:
            in_bounds          = (
                (uv[:, 0] >= 0) & (uv[:, 0] < width) &
                (uv[:, 1] >= 0) & (uv[:, 1] < height)
            )
            uv[~in_bounds]     = np.nan

        pixels[valid] = uv

    return pixels
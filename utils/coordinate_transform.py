import numpy as np

def local_2_global(location: np.ndarray, points: np.ndarray, rotation: float):

    x, y, z = location
    c, s = np.cos(rotation), np.sin(rotation)

    T = np.array([
        [ c, -s, x],
        [ s,  c, y],
        [ 0,  0, 1]
    ])

    pts = np.atleast_2d(points)
    pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
    local_pts = (T @ pts.T).T
    return local_pts if len(local_pts) > 1 else local_pts[0]


def global_2_local(location: np.ndarray, point: np.ndarray, rot: float):
    x, y, z = location
    c, s = np.cos(rot), np.sin(rot)

    T = np.array([
        [ c,  s, -x*c - y*s],
        [ s, -c, -x*s + y*c],
        [ 0,  0,        1 ]
    ])

    pts = np.atleast_2d(point)
    pts = np.hstack([pts[:, :2], np.ones((pts.shape[0], 1))])
    local_pts = (T @ pts.T).T
    return local_pts[:, :2] if len(local_pts) > 1 else local_pts[0, :2]
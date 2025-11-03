import numpy as np
from scipy.interpolate import interp1d


def lateral_control(waypoints: np.ndarray, Ld: float, wheelbase: float, max_steer: float):
    padded_wp = np.r_[np.zeros((1, 2)), waypoints]
    s  = np.linalg.norm(padded_wp, axis = 1) 
    xs = interp1d(s, padded_wp[:, 0], bounds_error = False, fill_value = (0, padded_wp[-1, 0]))
    ys = interp1d(s, padded_wp[:, 1], bounds_error = False, fill_value = (0, padded_wp[-1, 1]))

    target_x = xs(Ld)
    target_y = ys(Ld)
    phi = np.atan2(target_x, target_y)
    steer = np.degrees(np.atan2(2 * wheelbase * np.sin(phi), np.sqrt(target_x ** 2 + target_y ** 2)))

    # print(xs(Ld), ys(Ld), steer)


    steer = np.clip(steer, -max_steer, max_steer)
    normalized_steer = steer / max_steer * 1.8
    return normalized_steer
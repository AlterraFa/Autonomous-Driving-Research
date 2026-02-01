import time
import numpy as np
from scipy.interpolate import interp1d

nowtime = time.time()

def lateral_control(waypoints: np.ndarray, Ld: float, wheelbase: float, max_steer: float):
    padded_wp = np.r_[np.zeros((1, 2)), waypoints]
    s  = np.linalg.norm(padded_wp, axis = 1) 
    xs = interp1d(s, padded_wp[:, 0], bounds_error = False, fill_value = (0, padded_wp[-1, 0]))
    ys = interp1d(s, padded_wp[:, 1], bounds_error = False, fill_value = (0, padded_wp[-1, 1]))

    target_x = xs(Ld)
    target_y = ys(Ld)
    phi = np.atan2(target_x, target_y)
    steer = np.degrees(np.atan2(2 * wheelbase * np.sin(phi), np.sqrt(target_x ** 2 + target_y ** 2)))

    steer = np.clip(steer, -max_steer, max_steer)
    normalized_steer = steer / max_steer * 1.2
    return normalized_steer

def longitudinal_speed(waypoints, num_waypoints_to_average=3, time_step=0.2):
    """
    Calculates a smoother target speed by averaging over several waypoints.

    :param waypoints: A list or numpy array of [x, y] waypoints.
    :param num_waypoints_to_average: The number of waypoints to look ahead for averaging.
    :param time_step: The time interval between each predicted waypoint.
    :return: The target speed in meters per second (m/s).
    """
    global nowtime
    if waypoints is None or len(waypoints) < 2:
        return 0.0

    num_to_consider = min(len(waypoints) - 1, num_waypoints_to_average)

    path_length = 0.0
    for i in range(num_to_consider):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        path_length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
    total_time = num_to_consider * time_step

    if total_time == 0:
        return 0.0

    target_speed_ms = path_length / total_time
    
    if target_speed_ms < 10:
        target_speed_ms = 10
    
    return target_speed_ms

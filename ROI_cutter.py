import os, sys, subprocess
import cv2
import glob
import numpy as np

def copy_with_roi(src: str) -> str:
    dst = src + "_roi"
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        subprocess.run(["cp", "-r", src, dst], check=True)
    elif sys.platform == "win32":
        subprocess.run(["robocopy", src, dst, "/E"], check=True, shell=True)
    return dst

src = "./data/recording_20250905_231635_best_spatial"
dst = copy_with_roi(src)
paths = glob.glob(dst + "/images/*")

H, W, _    = 720, 1280, 3
x_top_left = 280; x_top_right = W - x_top_left
x_bot_left = 180; x_bot_right = W - x_bot_left
y_hor      = 390; y_bot         = 720
src_points = np.float32([[x_top_left, y_hor],
                        [x_top_right, y_hor],
                        [x_bot_right, y_bot],
                        [x_bot_left, y_bot]])
width = 250; height = 150
dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

M = cv2.getPerspectiveTransform(src_points, dst_points)

for path in paths:
    img = cv2.imread(path)
    warped = cv2.warpPerspective(img, M, (width, height))
    cv2.imwrite(path, warped)
    # cv2.imshow("test", warped)
    # key = cv2.waitKey(0)
    # if key == ord("q"):
    #     break
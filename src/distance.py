import numpy as np
import cv2
import os
import time
import sys
from numba import jit

def read_npy_file(path, idx):
    img_depth = np.load(os.path.join(path, "frame_depth_" + str(idx) + ".npy"))
    img_color = np.load(os.path.join(path, "frame_color_" + str(idx) + ".npy"))
    return (img_depth, img_color)


def play_video_from_npy(path):
    """
    Play numpy array files along with the chosen preprocessing.
    Shapes:
        img_color: (480, 640, 3), dtype: np.uint8
        img_depth: (480, 640), dtpye: np.uint16
        colormap_depth: (480, 640, 3), dtype: np.uint8
    """
    datatype = np.uint8
    sleep_time = 0.01
    idx = 0
    try:
        while(True):
            (img_depth, img_color) = read_npy_file(path, idx)
            
            idx += 1

            colormap_depth = apply_colormap(img_depth)
            print("Depth: {}".format(colormap_depth.shape))

            #* Put processing function here.

            # display
            cv2.imshow("color_image", img_color.astype(datatype))
            cv2.imshow("depth_image", colormap_depth.astype(datatype))
            
            # required for proper display
            if cv2.waitKey(1) == ord('q'): break
            time.sleep(sleep_time)           
    except:
        print("[INFO] Video sequence end.")
    finally:
        cv2.destroyAllWindows()


def apply_colormap(img_depth):
    return cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), 9)

if __name__ == '__main__':
    path = "./meresek/rec0/"
    play_video_from_npy(path)
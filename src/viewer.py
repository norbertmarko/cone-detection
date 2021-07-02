import numpy as np
import cv2
import time
import sys
import os

num = 0
path = ""
if (len(sys.argv) == 2):
    path = sys.argv[1]

sleep_time = 0.01

try:
    while (True):
        depth_image = np.load(os.path.join(path, "frame_depth_" + str(num) + ".npy"))
        color_image = np.load(os.path.join(path, "frame_color_" + str(num) + ".npy"))

        num = num + 1

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), 9)

        cv2.imshow("color_image", color_image.astype(np.uint8))
        cv2.imshow("depth_image", depth_colormap)
        
        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(sleep_time)
finally:
    cv2.destroyAllWindows() 
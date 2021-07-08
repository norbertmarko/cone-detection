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
        while True:
            (img_depth, img_color) = read_npy_file(path, idx)
            
            idx += 1

            colormap_depth = apply_colormap(img_depth)

            #* Put processing function here.
            (img_depth, img_color) = testfunc(img_depth, img_color)

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


#TODO: convert numpy files to .mp4 to calibrate HSV for it


def testfunc(img_depth, img_color):
    """
    Function with experiments.
    """

    #TODO: filter area on color image, collect depth 
    #values from filtered area,than give back final depth by taking average and min dist

    frame_HSV = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    # (low_H, low_S, low_V), (high_H, high_S, high_V)
    frame_thresholded = cv2.inRange(frame_HSV, (0, 70, 171), (60, 255, 255))

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(frame_thresholded, cv2.MORPH_OPEN, kernel)
    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
    
    # edge and contour
    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #max_contour = max(contours, key=cv2.contourArea)
    img_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)

    # max_contour borders the area we need

    # approx. contours
    approx_contours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    # video ckpt
    img_approx_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

    # convex hulls
    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))

    img_all_convex_hulls = np.zeros_like(img_edges)
    cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)

    return (img_depth, img_all_convex_hulls)


if __name__ == '__main__':
    path = "./meresek/rec1/"
    play_video_from_npy(path)
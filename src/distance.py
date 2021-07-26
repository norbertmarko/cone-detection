from textwrap import fill
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


            polygon = create_poly(img_color)
            filtered_depth, min_dist, avg_dist = get_cone_depth(img_depth, polygon)
            center = get_cone_center(polygon)

            
            print(f"Current center pixel (x,y): {center}")
            print(f"Current minimum distance: {min_dist}")
            print(f"Current average distance: {avg_dist}")

            # display
            cv2.imshow("color_image", polygon.astype(datatype))
            cv2.imshow("depth_image", filtered_depth.astype(datatype))
            
            # required for proper display
            if cv2.waitKey(1) == ord('q'): break
            time.sleep(sleep_time)           
    except:
        print("[INFO] Video sequence end.")
    finally:
        cv2.destroyAllWindows()


def apply_colormap(img_depth):
    return cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), 9)


def create_poly(img_color):
    """
    Create a polygon around the traffic cone. 
    """

    frame_HSV = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    # (low_H, low_S, low_V), (high_H, high_S, high_V)
    frame_thresholded = cv2.inRange(frame_HSV, (0, 100, 171), (180, 255, 255))

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(frame_thresholded, cv2.MORPH_OPEN, kernel)
    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
    
    # edge and contour
    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)

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
    hull_contours, _ = cv2.findContours(img_all_convex_hulls, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    max_hull_contour = max(hull_contours, key=cv2.contourArea)
    max_hull_contour_img = np.zeros_like(img_edges)
    cv2.drawContours(
        max_hull_contour_img, max_hull_contour, -1, (255,255,255), 2, cv2.FILLED
    )
    polygon = np.zeros_like(img_edges)
    cv2.fillPoly(polygon, pts =[max_hull_contour], color=(255,255,255))

    return polygon


def get_cone_depth(img_depth, polygon):
    # create a boolean mask (if the value is less than 255 -> True) 
    mask = polygon < 255
    # set img_depth to 0 where the mask is True (shapes match)
    img_depth[mask] = 0

    min_dist = np.max(img_depth[:, :])
    avg_dist = np.average(img_depth[:, :])

    return (img_depth, min_dist, avg_dist)


def get_cone_center(polygon):
    """
    Returns traffic cone center 
    on the image (x, y) from the
    minimum enclosing circle.
    """
    contours, _ = cv2.findContours(
        polygon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = contours[0]
    (x, y), r = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(r)

    return center


if __name__ == '__main__':
    path = "./meresek/rec1/"
    play_video_from_npy(path)
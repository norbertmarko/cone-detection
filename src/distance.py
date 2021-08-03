from textwrap import fill
import numpy as np
import cv2 as cv
import os
import time
import sys
import imutils
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
    idx = 1
    try:
        while True:
            (img_depth, img_color) = read_npy_file(path, idx)
            
            idx += 1

            colormap_depth = apply_colormap(img_depth)

            #* Put processing function here.

            try:
                polygon, blurred = create_poly(img_color)
                filtered_depth, min_dist, avg_dist = get_cone_depth(img_depth, polygon)
                center = get_cone_center(polygon)
				
                
                print(f"Current center pixel (x,y): {type(center[0])}")
                print(f"Current minimum distance: {type(min_dist)}")
                print(f"Current average distance: {type(avg_dist)}")

                # display
                cv.imshow("original_color",img_color.astype(datatype))
                cv.imshow("filtered_color",blurred.astype(datatype))
                cv.imshow("polygon", polygon.astype(datatype))
                cv.imshow("depth_image", filtered_depth.astype(datatype))
            except:
                pass
            
            # required for proper display
            if cv.waitKey(1) == ord('q'): break
            time.sleep(sleep_time)           
    except Exception as e:
        print(e)
        print("[INFO] Video sequence end.")
    finally:
        cv.destroyAllWindows()


def apply_colormap(img_depth):
    return cv.applyColorMap(cv.convertScaleAbs(img_depth, alpha=0.03), 9)

def create_poly(image):

	# first array: x >= , second array: x <= (B, G, R)
	boundaries = ([0, 0, 200], [121, 151, 255])

	lower, upper = boundaries
	lower = np.array(lower, dtype=np.uint8)
	upper = np.array(upper, dtype=np.uint8)

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(image, lower, upper)
	output = cv.bitwise_and(image, image, mask= mask)
	
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
	closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)
	blurred = cv.medianBlur(closing, 5)
	img_edges = cv.Canny(blurred, 30, 160)
	cnts, hierarcy = cv.findContours(img_edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	contours = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
	polygon = np.zeros_like(img_edges)
	cv.drawContours(polygon, contours, -1, (255, 255, 255), cv.FILLED)
	return polygon, blurred
    

# def create_poly(image):

#     # first array: x >= , second array: x <= (B, G, R)
#     boundaries = ([0, 0, 200], [121, 151, 255])

#     lower, upper = boundaries
#     lower = np.array(lower, dtype=np.uint8)
#     upper = np.array(upper, dtype=np.uint8)

#     # find the colors within the specified boundaries and apply
#     # the mask
#     mask = cv.inRange(image, lower, upper)
#     output = cv.bitwise_and(image, image, mask= mask)

#     # postprocessing operations
#     kernel_sizes = [(3, 3), (5, 5), (7, 7)]
#     size = kernel_sizes[1]
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
#     closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)

#     #TODO: try bilateral blur - reduces noise while preserving edges
#     blurred = cv.medianBlur(closing, 5)
#     # edge detection
#     img_edges = cv.Canny(blurred, 30, 160)
#     # find contours (segments?)
#     cnts = cv.findContours(img_edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sorted(cnts, key = cv.contourArea, reverse=True)

#     img_contours = np.zeros_like(img_edges)
#     cv.drawContours(img_contours, cnts, -1, (255,255,255), 3)
#     polygon = np.zeros_like(img_edges)
#     cv.fillPoly(polygon, pts =[cnts[0]], color=(255,255,255))
    
#     return polygon, blurred


def old_poly(img_color):
    """
    Create a polygon around the traffic cone. 
    """

    frame_HSV = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    # (low_H, low_S, low_V), (high_H, high_S, high_V)
    frame_thresholded = cv.inRange(frame_HSV, (0, 100, 171), (180, 255, 255))

    kernel = np.ones((5, 5))
    img_thresh_opened = cv.morphologyEx(frame_thresholded, cv.MORPH_OPEN, kernel)
    img_thresh_blurred = cv.medianBlur(img_thresh_opened, 5)
    
    # edge and contour
    img_edges = cv.Canny(img_thresh_blurred, 80, 160)
    contours, _ = cv.findContours(np.array(img_edges), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(img_edges)
    cv.drawContours(img_contours, contours, -1, (255,255,255), 2)

    # approx. contours
    approx_contours = []

    for c in contours:
        approx = cv.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    # video ckpt
    img_approx_contours = np.zeros_like(img_edges)
    cv.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

    # convex hulls
    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv.convexHull(ac))


    img_all_convex_hulls = np.zeros_like(img_edges)
    cv.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)
    hull_contours, _ = cv.findContours(img_all_convex_hulls, cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    max_hull_contour = max(hull_contours, key=cv.contourArea)
    max_hull_contour_img = np.zeros_like(img_edges)
    cv.drawContours(
        max_hull_contour_img, max_hull_contour, -1, (255,255,255), 2, cv.FILLED
    )
    polygon = np.zeros_like(img_edges)
    cv.fillPoly(polygon, pts =[max_hull_contour], color=(255,255,255))

    return polygon


def get_cone_depth(img_depth, polygon):
    # create a boolean mask (if the value is less than 255 -> True) 
    mask = polygon < 255
    # set img_depth to 0 where the mask is True (shapes match)
    img_depth[mask] = 0

    min_dist = np.max(img_depth[:, :])
    avg_dist = np.average(img_depth[:, :])

    return (img_depth, np.float32(min_dist), np.float32(avg_dist))


def get_cone_center(polygon):
    """
    Returns traffic cone center 
    on the image (x, y) from the
    minimum enclosing circle.
    """
    contours, _ = cv.findContours(
        polygon, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    cnt = contours[0]
    (x, y), r = cv.minEnclosingCircle(cnt)
    center = [np.int32(x), np.int32(y)]
    radius = int(r)

    return center


if __name__ == '__main__':
    path = "./meresek/rec1/"
    play_video_from_npy(path)
import numpy as np
import cv2


def convex_hull_pointing_up(ch):
    '''Determines if the path is pointing up.
    If so, this is a cone.'''
        
    # contour points above center and below
    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv2.boundingRect(ch) # coordinates of the upper-left corner of the enclosing rectangle, width and height
    aspect_ratio = w / h # ratio of rectangle width to height

    # if the rectangle is narrow, continue defining. If not, then the circuit does not fit
    if aspect_ratio < 0.8:
        # each point of the contour is classified as lying above or below the center
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center: # if the y-coordinate of the point is above the center, then add this point to the list of points above the center
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        # define the x coordinates of the extreme points below the center
        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        # check if the top points of the path are outside the "base". If yes, then the circuit does not fit
        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False
        
    return True


cap = cv2.VideoCapture("test_videos/20210326_120239.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1] / 2.0), int(frame.shape[0] / 2.0)))

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # (low_H, low_S, low_V), (high_H, high_S, high_V)
    frame_thresholded = cv2.inRange(frame_HSV, (0, 70, 171), (60, 255, 255))

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

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)

    # cones = []
    # bounding_rects = []
    # for ch in convex_hulls_3to10:
    #     if convex_hull_pointing_up(ch):
    #         cones.append(ch)
    #         rect = cv2.boundingRect(ch)
    #         bounding_rects.append(rect)

    # img_cones = np.zeros_like(img_edges)
    # cv2.drawContours(img_cones, cones, -1, (255,255,255), 2)
    # cv2.drawContours(img_cones, bounding_boxes, -1, (1,255,1), 2)

    cv2.imshow("frame", img_convex_hulls_3to10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np


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

    cv2.imshow("frame", img_convex_hulls_3to10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import argparse
from scipy.spatial import distance

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='v0.mp4')
args = parser.parse_args()


#cap = cv2.VideoCapture(args.video)
cap = cv2.VideoCapture("test_videos/20210326_120239.mp4")

idx = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (int(frame.shape[1] / 2.0), int(frame.shape[0] / 2.0)))

    r, g, b = frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]

    mask = (r > 200) * (g < 150) * (b < 150)
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5))
    mask = cv2.dilate(mask, kernel, iterations = 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    img_hull = np.zeros(frame.shape).astype(np.uint8)
    maxarea = 0
    maxareaidx = -1
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        area = cv2.contourArea(hull)
        if area > maxarea:
            maxarea = area
            maxareaidx = i
        hull_list.append(hull)

    cv2.drawContours(img_hull, hull_list, maxareaidx, (0, 255, 0), 3)
    minx = 0
    minxidx = -1
    for p in range(len(hull_list[maxareaidx])):
        if hull_list[maxareaidx][p][0][1] > minx:
            minx = hull_list[maxareaidx][p][0][1]
            minxidx = p

    c = np.array((hull_list[maxareaidx][minxidx][0][0], hull_list[maxareaidx][minxidx][0][1]))

    img_hull = cv2.cvtColor(img_hull, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(img_hull,1,np.pi/180,100)
    img_hull = np.zeros(frame.shape).astype(np.uint8)
    
    mindist0 = 1000000
    mindist1 = 1000000
    mindistidx0 = -1
    mindistidx1 = -1

    if not (lines is None): 
        nplines = np.squeeze(np.array(lines))
        for i in range(int(len(lines))):
            dist0 = distance.euclidean(c, nplines[i, :2])
            dist1 = distance.euclidean(c, nplines[i, 2:])
            if dist0 < mindist0:
                mindist0 = dist0
                mindistidx0 = i
            if dist1 < mindist1:
                mindist1 = dist1
                mindistidx1 = i
    
    cv2.line(frame,(nplines[mindistidx0][0], nplines[mindistidx0][1]), (nplines[mindistidx0][2], nplines[mindistidx0][3]),(255,0,0),4)
    cv2.line(frame,(nplines[mindistidx1][0], nplines[mindistidx1][1]), (nplines[mindistidx1][2], nplines[mindistidx1][3]),(255,0,0),4)
    cv2.circle(frame, (c[0], c[1]), 10, (255, 0, 255), 4)
    
    #cv2.imwrite(args.video + "_" + str(idx) + ".png", frame)
    #idx += 1

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

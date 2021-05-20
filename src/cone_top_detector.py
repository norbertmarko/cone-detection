#!/usr/bin/env python

from timeit import default_timer as timer

import cv2
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ConeDetectionTopROSNode():

    def __init__(self):
        self.CvBridge = CvBridge()
        self.Subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.callback, queue_size=1)
        self.ConeDetectionPublisher = rospy.Publisher("detected_cone", Image, queue_size=1)
        self.ConeEdgePublisher = rospy.Publisher("detected_cone_edges", Image, queue_size=1)        
        print("[INIT OK]")       

    def callback(self, ros_msg):
        start_time = timer()

        frame = self.CvBridge.imgmsg_to_cv2(ros_msg, "bgr8")
        
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_thresholded = cv2.inRange(frame_HSV, (0, 70, 171), (60, 255, 255))
        kernel = np.ones((5, 5))
        img_thresh_opened = cv2.morphologyEx(frame_thresholded, cv2.MORPH_OPEN, kernel)
        img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
        img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

        img_edges = cv2.dilate(img_edges, kernel, iterations = 5)

        contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hulled = cv2.convexHull(contours[0])

        img_hull = frame.copy()
        cv2.drawContours(img_hull, [hulled], 0, (0, 255, 0), -1)

        masked_frame = ((frame.astype(float) + img_hull.astype(float)) / 2).astype(np.uint8)        

        image_msg = self.CvBridge.cv2_to_imgmsg(masked_frame, encoding="bgr8")
        image_msg.header = Header(frame_id="cone_detection_output", stamp=ros_msg.header.stamp)
        self.ConeDetectionPublisher.publish(image_msg)

        img_hull = np.zeros(frame.shape).astype(np.uint8)
        cv2.drawContours(img_hull, [hulled], 0, (0, 255, 0), 2)
        img_hull = cv2.cvtColor(img_hull, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(img_hull, 1, np.pi/180, 100)
        for i in range(lines.shape[0]):
            cv2.line(frame, (lines[i, 0, 0], lines[i, 0, 1]), (lines[i, 0, 2], lines[i, 0, 3]), (255, 0, 0), 10)

        image_msg = self.CvBridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header = Header(frame_id="cone_detection_output", stamp=ros_msg.header.stamp)
        self.ConeEdgePublisher.publish(image_msg)
        
        print("[Max. FPS] ", (1.0 / (timer() - start_time)))

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node("cone_detector_top", anonymous=True)
    tf_seg_ros = ConeDetectionTopROSNode()
    tf_seg_ros.main()

from typing import Tuple

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image

#TODO: integrate code with pathlib
import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2 as cv


class ROSPublisher(Node):

    def __init__(self):
        super().__init__('computer_vision_publisher')
        self.pipeline = None
        self.align = None
        self.calibrate_camera()

        #TODO: correct publishers
        # depth publisher (image)
        self.depth_image_publisher_ = self.create_publisher(Image, 'cv/depth_image_cone', 10)
        # minimum distance publisher (int)
        self.min_dist_publisher_ = self.create_publisher(String, 'cv/min_dist_cone', 10)
        # average distance publisher (int)
        self.avg_dist_publisher_ = self.create_publisher(String, 'cv/avg_dist_cone', 10)
        # cone center point publisher (tuple: int)
        self.cone_center_publisher_ = self.create_publisher(String, 'cv/center_cone', 10)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0


    # def read_npy_frame(self, path: str, idx: int) -> Tuple[np.array, np.array]:
    #     """
    #     Loads numpy array frame from an .npy file with the given index value.
    #     """
    #     img_depth = np.load(os.path.join(path, "frame_depth_" + str(idx) + ".npy"))
    #     img_color = np.load(os.path.join(path, "frame_color_" + str(idx) + ".npy"))

    #     return (img_depth, img_color)

    def calibrate_camera(self) -> None:
        """
        Calibrates RealSense camera. Called in the constructor.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The ROS Node requires Depth camera with Color sensor.")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def create_poly(self, img_color: np.array) -> np.array:
        """
        Create a polygon around the traffic cone. 
        """
        #TODO: optimize function, filter unneccessary operations 
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

    
    def filter_depth_img(self, img_depth: np.array, poly: np.array) -> np.array:
        """
        Sets depth image pixels which do not
        belong to the cone (polygon) to 0.
        """
        mask = poly < 255
        img_depth[mask] = 0

        return img_depth

    
    def calc_distance(self, img_depth: np.array) -> Tuple[float, float]:
        """
        Returns the cone's minimal and average distance.
        """
        min_dist = np.max(img_depth[:, :])
        avg_dist = np.average(img_depth[:, :])

        return (min_dist, avg_dist)


    def get_cone_center(polygon: np.array) -> Tuple[int, int]:
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
        center = (int(x), int(y))
        radius = int(r)

        return center


    def timer_callback(self) -> None:
        
        try:
            # Read RealSense data
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            img_depth = np.asanyarray(
                aligned_frames.get_depth_frame().get_data()
            )
            img_color = np.asanyarray(
                aligned_frames.get_color_frame().get.data()
            )

            polygon = self.create_poly(img_color)
            filtered_depth = self.filter_depth_img(img_depth, polygon)
            min_dist, avg_dist = self.calc_distance(filtered_depth)
            center = self.get_cone_center(polygon)

            # Publish calculations

            # pred = np.uint8(colors[pred])
            # image_msg = self.CvBridge.cv2_to_imgmsg(pred, encoding="bgr8")
            # image_msg.header = Header(frame_id="seg_output",stamp=rospy.Time.now())
            # self.publisher.publish(image_msg)



        except:
            print("[INFO] Video sequence terminated.")
        finally:
            cv.destroyAllWindows()


        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    minimal_publisher = ROSPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import sys
from typing import Tuple
# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, Float64, Int32MultiArray
from sensor_msgs.msg import Image
# Computer Vision Imports
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import math
from math import sin, cos


class ROSPublisher(Node):
    
    def __init__(self, timer_period=0.1):
        super().__init__('cone_ori_publisher')
        self.timer_period = timer_period
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.br = CvBridge()
        # calibration function
        self.pipeline = None
        self.align = None
        self.calibrate_camera()
        # ArUco stuff
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        # cone orientation publisher (float32)
        self.cone_orientation_publisher = self.create_publisher(
            Float64, 'cv/ori', 10
        )
        # cone center point publisher (list: int32)
        self.cone_center_publisher = self.create_publisher(
            Int32MultiArray, 'cv/xy', 10
        )
        # cone orientation visualizer (image)
        self.cone_image_publisher = self.create_publisher(
            Image, 'cv/vis', 10
        )


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


    def timer_callback(self) -> None:
        """
        Main callback function for ROS2.
        This function gets called every 0.1 seconds
        (set by timer).
        """
        try:
            # Read RealSense data
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_image = np.asanyarray(
                aligned_frames.get_depth_frame().get_data()
            )
            color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
            try:
                (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, arucoDict4, parameters=arucoParams)
                if len(corners) > 0:
                    # flatten the ArUco IDs list
                    ids = ids.flatten()

                    # loop over the detected ArUco corners
                    for (markerCorner, markerID) in zip(corners, ids):
                        # extract the marker corners (which are always returned in
                        # top-left, top-right, bottom-right, and bottom-left order)
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners

                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))

                        # compute the center (x, y)-coordinates of the ArUco marker
                        center_x = int((topLeft[0] + bottomRight[0]) / 2.0)
                        center_y = int((topLeft[1] + bottomRight[1]) / 2.0)

                        # compute the top side orientation of the ArUci marker 
                        orientation = degrees(atan2(abs(topLeft[0] - topRight[0]), abs(topLeft[1] - topRight[1])))

                        (x0, y0), (x1, y1) = topLeft, topRight
                        
                        if x0 >= x1 and y0 <= y1:
                            orientation = 360 - orientation
                        
                        elif x0 >= x1 and y0 >= y1:
                            orientation = 180 + orientation

                        elif x0 <= x1 and y0 >= y1:
                            orientation = 180 - orientation

                        sx = int((x0 + x1) / 2.0)
                        sy = int((y0 + y1) / 2.0)

                        theta = np.deg2rad(orientation - 180)
                        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                        v = np.array([0, 200])
                        v2 = np.dot(rot, v)

                        cv2.arrowedLine(color_image, (sx, sy), (int(sx + v2[1]), int(sy + v2[0])), (255, 0, 255), 3)
                        cv2.putText(color_image, str(int(orientation)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

                        img_msg = self.br.cv2_to_imgmsg(np.uint8(color_image), encoding='passthrough')
                        img_msg.header = Header(frame_id="vis", stamp=self.get_clock().now().to_msg())
                        self.cone_image_publisher.publish(img_msg)

                        ori_msg = Float64()
                        ori_msg.data = orientation
                        self.cone_orientation_publisher.publish(ori_msg)

                        cone_center_msg = Int32MultiArray()
                        cone_center_msg.data = [center_x, center_y]
                        self.cone_center_publisher.publish(cone_center_msg)

            except Exception as e:
                print(
                    '[INFO] Error (Exception) on line {}'.format(
                    sys.exc_info()[-1].tb_lineno), type(e).__name__, e
                )

        except Exception as e:
            print("[INFO] Exception cause: %s" % e)
            print("[INFO] Video sequence terminated.")
        finally:
            cv.destroyAllWindows()


def main(args=None) -> None:
    rclpy.init(args=args)
    publisher = ROSPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
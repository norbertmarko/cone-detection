from typing import Tuple
# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
# Computer Vision Imports
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2 as cv
import numpy as np


class ROSPublisher(Node):
    
    def __init__(self, timer_period=0.1):
        super().__init__('computer_vision_publisher')
        self.timer_period = timer_period
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.br = CvBridge()
        # calibration function
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
        #TODO: streamline function
        min_dist = np.max(img_depth[:, :])
        avg_dist = np.average(img_depth[:, :])

        return (min_dist, avg_dist)


    def get_cone_center(polygon: np.array) -> Tuple[int, int]:
        """
        Returns traffic cone center 
        on the image (x, y) from the
        minimum enclosing circle.
        """
        #TODO: filter down function
        contours, _ = cv.findContours(
            polygon, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        cnt = contours[0]
        (x, y), r = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)

        return center


    def calc_cone_polygon(self, img_color: np.array) -> np.array:
        """
        Calculates polygon around the traffic cone. 
        """
        polygon = np.zeros((2, 2), dtype=np.uint8)
        return polygon


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
            img_depth = np.asanyarray(
                aligned_frames.get_depth_frame().get_data()
            )
            img_color = np.asanyarray(
                aligned_frames.get_color_frame().get.data()
            )

            # callback content (run functions here)
            
            #self.publisher.publish(self.br.cv2_to_imgmsg(frame))
            #self.get_logger().info('Publishing frame')

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
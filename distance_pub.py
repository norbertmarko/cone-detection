from typing import Tuple
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
#TODO: integrate code with pathlib
import os

# https://automaticaddison.com/getting-started-with-opencv-in-ros-2-foxy-fitzroy-python/#Create_the_Image_Publisher_Node_Python

# parameters
node_name = 'distance_publisher'
topic_name = 'cone_distance'
timer_period = 0.1


class Publisher(Node):
    def __init__(self):
        super().__init__(node_name)
        self.publisher = self.create_publisher(Image, topic_name, 10)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.br = CvBridge()        

    #! subscribe to polygon node
    #! publish cone_depth
    #? Multiple publishers in one node?

    def filter_depth_img(img_depth: np.array, polygon: np.array) -> np.array:
        """
        Sets depth image pixels which do not belong to the cone to 0.
        """
        mask = polygon < 255
        img_depth[mask] = 0

        return img_depth


    def calc_distance(img_depth: np.array) -> Tuple[float, float]:
        """
        Returns the cone's minimal and average distance.
        """
        min_dist = np.max(img_depth[:, :])
        avg_dist = np.average(img_depth[:, :])

        return (min_dist, avg_dist)

        

    def timer_callback(self) -> None:
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        frame = None
        
        self.publisher.publish(self.br.cv2_to_imgmsg(frame))
        self.get_logger().info('Publishing frame')


def main(args=None) -> None:
    rclpy.init(args=args)
    publisher = Publisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


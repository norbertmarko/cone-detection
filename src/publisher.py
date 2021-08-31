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
		# cone center point publisher (list: int32)
		self.cone_center_publisher_ = self.create_publisher(
			Int32MultiArray, 'cv/center_cone', 10
		)
		# minimum distance publisher (float64)
		self.min_dist_publisher_ = self.create_publisher(
			Float64, 'cv/min_dist_cone', 10
		)
		# average distance publisher (float64)
		self.avg_dist_publisher_ = self.create_publisher(
			Float64, 'cv/avg_dist_cone', 10
		)
		# depth publisher (image)
		self.depth_image_publisher_ = self.create_publisher(
			Image, 'cv/depth_image_cone', 10
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
		#TODO: streamline function, implement more precise calculations
		min_dist = np.min(img_depth[np.nonzero(img_depth)])
		avg_dist = np.average(img_depth[np.nonzero(img_depth)])

		return (float(min_dist), float(avg_dist))


	def get_cone_center(self, polygon: np.array) -> Tuple[int, int]:
		"""
		Returns traffic cone center 
		on the image (x, y) from the
		minimum enclosing circle.
		"""
		#TODO: implement a better solution
		contours, _ = cv.findContours(
			polygon, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
		)
		cnt = contours[0]
		(x, y), r = cv.minEnclosingCircle(cnt)
		center = [int(x), int(y)]
		radius = int(r)

		return center

	def visualize_cone_center(self, img_color: np.array, polygon: np.array, center: Tuple[int, int]) -> np.array:
		"""
		Visualizes the cone center on the color image.
		"""
		contours, _ = cv.findContours(
			polygon.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
		)
		# cnt = contours[0]
		for c in contours:

			# draw the contour and center of the shape on the image
			cv.drawContours(img_color, [c], -1, (0, 255, 0), 2)
			cv.circle(img_color, (center[0], center[1]), 7, (0, 255, 0), -1)
			cv.putText(img_color, "center", (center[0] - 20, center[1] - 20),
				cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		return img_color


	def calc_cone_polygon(self, img_color: np.array) -> np.array:
		"""
		Calculates polygon around the traffic cone. 
		"""
		#TODO: Histogram matching.
		# Boundaries
		lower, upper = (
			np.array([0, 0, 200], dtype=np.uint8), 
			np.array([121, 200, 255], dtype=np.uint8)
		)
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv.inRange(img_color, lower, upper)
		output = cv.bitwise_and(img_color, img_color, mask=mask)
		# prepare image
		kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
		closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)
		blurred = cv.medianBlur(closing, 5)
		# edges and contours
		img_edges = cv.Canny(blurred, 30, 160)
		cnts, _ = cv.findContours(
			img_edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
		)
		sorted_cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
		# hull calculation
		hull = []
		for i in range(len(sorted_cnts)):
			hull.append(cv.convexHull(sorted_cnts[i], False))
		polygon = np.zeros_like(img_edges)
		cv.drawContours(
			polygon, hull, -1, (255, 255, 255), cv.FILLED
		)        
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
				aligned_frames.get_color_frame().get_data()
			)

			try:
				# Process frame
				polygon = self.calc_cone_polygon(img_color)
				img_depth_filtered = self.filter_depth_img(img_depth, polygon)
				min_dist, avg_dist = self.calc_distance(img_depth_filtered)
				center = self.get_cone_center(polygon)

				# Print results (debug)
				img_color = self.visualize_cone_center(img_color, polygon, center)
				print(f"Current center pixel (x,y): {center}")
				print(f"Current minimum distance: {min_dist / 1000} m")
				print(f"Current average distance: {avg_dist / 1000} m")

				# Publish data
				cone_center_msg = Int32MultiArray()
				cone_center_msg.data = center

				min_dist_msg = Float64()
				min_dist_msg.data = min_dist / 1000

				avg_dist_msg = Float64()
				avg_dist_msg.data = avg_dist / 1000

				#TODO: refactor to the same format as above
				depth_img_msg = self.br.cv2_to_imgmsg(
					np.uint8(img_color), encoding='bgr8'
				)
				depth_img_msg.header = Header(
					frame_id="depth", stamp=self.get_clock().now().to_msg()
				)

				self.cone_center_publisher_.publish(cone_center_msg)
				self.min_dist_publisher_.publish(min_dist_msg)
				self.avg_dist_publisher_.publish(avg_dist_msg)
				self.depth_image_publisher_.publish(depth_img_msg)
				self.get_logger().info('Publishing...')
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
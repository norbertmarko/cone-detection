import sys
from typing import Tuple
# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, Float64, Int32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
# Computer Vision Imports
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import math
from math import sin, cos, degrees, atan2

class ROSPublisher(Node):
	
	def __init__(self, timer_period=0.1):
		super().__init__('computer_vision_publisher')

		self.points = np.zeros((307200, 7))
		self.cone_point = np.zeros((1, 7))

		self.timer_period = timer_period
		self.timer = self.create_timer(self.timer_period, self.timer_callback)
		self.br = CvBridge()
		# calibration function
		self.pipeline = None
		self.align = None
		self.calibrate_camera()
		
		#### ORIENTATION
		# ArUco stuff
		self.arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
		self.arucoParams = cv.aruco.DetectorParameters_create()
		# cone orientation publisher (float32)
		self.cone_orientation_publisher = self.create_publisher(
			Float64, 'cv/top_orientation_cone', 10
		)
				# cone orientation visualizer (image)
		self.cone_image_publisher = self.create_publisher(
			Image, 'cv/top_visualization_cone', 10
		)
		# cone center point publisher (list: int32)
		self.cone_center_publisher = self.create_publisher(
			Int32MultiArray, 'cv/center_top_cone', 10
		)
		####	

		# cone center point publisher (list: int32)
		self.cone_center_publisher_ = self.create_publisher(
			Int32MultiArray, 'cv/center_side_cone', 10
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
		self.pcd_all_publisher = self.create_publisher(PointCloud2, 'cv/all_pcd', 10)
		self.pcd_cone_publisher = self.create_publisher(PointCloud2, 'cv/cone_pcd', 10)


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
		self.intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()


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
		if len(contours) == 0:
			return (-1, -1)

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
			depth_frame = aligned_frames.get_depth_frame().as_depth_frame()

			try:
				#### ORIENTATION
				orientation = -1.0
				center_x = -1
				center_y = -1

				(corners, ids, rejected) = cv.aruco.detectMarkers(img_color, self.arucoDict, parameters=self.arucoParams)
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

						sideLength = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

						print(sideLength)
						
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
						v = np.array([0, sideLength * 4])
						v2 = np.dot(rot, v)

						cv.arrowedLine(img_color, (sx, sy), (int(sx + v2[1]), int(sy + v2[0])), (255, 0, 0), 5)
						cv.circle(img_color, (center_x, center_y), int(sideLength / 3.0), (0, 255, 0), -1)
						cv.putText(img_color, str(int(orientation)) + ", (" + str(center_x) + ", " + str(center_y) + ")", (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)


				print(orientation, center_x, center_y)


				num = 0

				for i in range(0, img_color.shape[0], 10):
					for j in range(0, img_color.shape[1], 10):
						position = rs.rs2_deproject_pixel_to_point(self.intr, (j, i), depth_frame.get_distance(j, i))
						self.points[num, 0] = position[0]
						self.points[num, 1] = position[1]
						self.points[num, 2] = position[2]
						self.points[num, 3] = img_color[i, j, 2] / 255.0
						self.points[num, 4] = img_color[i, j, 1] / 255.0
						self.points[num, 5] = img_color[i, j, 0] / 255.0
						self.points[num, 6] = 0.5
						num += 1
				
				self.pcd = point_cloud(self.points, 'map')
				self.pcd_all_publisher.publish(self.pcd)


				####
				
				# Process frame
				polygon = self.calc_cone_polygon(img_color)
				img_depth_filtered = self.filter_depth_img(img_depth, polygon)
				min_dist, avg_dist = 0, 0 #self.calc_distance(img_depth_filtered)
				center = self.get_cone_center(polygon)


				# Print results (debug)
				img_color = self.visualize_cone_center(img_color, polygon, center)
				print(f"Current center pixel (x,y): {center}")
				print(f"Current minimum distance: {min_dist / 1000} m")
				print(f"Current average distance: {avg_dist / 1000} m")

				# Publish data

				#### ORIENTATION
				img_msg = self.br.cv2_to_imgmsg(np.uint8(img_color), encoding='bgr8')
				img_msg.header = Header(frame_id="vis", stamp=self.get_clock().now().to_msg())
				self.cone_image_publisher.publish(img_msg)

				ori_msg = Float64()
				ori_msg.data = orientation
				self.cone_orientation_publisher.publish(ori_msg)

				cone_center_msg = Int32MultiArray()
				cone_center_msg.data = [center_x, center_y]
				self.cone_center_publisher.publish(cone_center_msg)
				####

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

				if not(center[0] == -1):
					position = rs.rs2_deproject_pixel_to_point(self.intr, center, depth_frame.get_distance(center[0], center[1]))
					self.cone_point[0, 0] = position[0]
					self.cone_point[0, 1] = position[1]
					self.cone_point[0, 2] = position[2]
					self.cone_point[0, 3] = 0.0
					self.cone_point[0, 4] = 1.0
					self.cone_point[0, 5] = 0.0
					self.cone_point[0, 6] = 0.0

					self.pcd = point_cloud(self.cone_point, 'map')
					self.pcd_cone_publisher.publish(self.pcd)
				else:
					self.pcd = point_cloud(np.zeros((0, 3)), 'map')
					self.pcd_cone_publisher.publish(self.pcd)

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

def point_cloud(points, parent_frame):
	ros_dtype = sensor_msgs.PointField.FLOAT32
	dtype = np.float32
	itemsize = np.dtype(dtype).itemsize

	data = points.astype(dtype).tobytes() 

	fields = [sensor_msgs.PointField(
		name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
		for i, n in enumerate('xyzrgba')]

	header = std_msgs.Header(frame_id=parent_frame)

	return sensor_msgs.PointCloud2(
		header=header,
		height=1, 
		width=points.shape[0],
		is_dense=False,
		is_bigendian=False,
		fields=fields,
		point_step=(itemsize * 7),
		row_step=(itemsize * 7 * points.shape[0]),
		data=data
	)


def main(args=None) -> None:
	rclpy.init(args=args)
	publisher = ROSPublisher()
	rclpy.spin(publisher)
	publisher.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

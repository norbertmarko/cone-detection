from typing import Tuple
import argparse
from pathlib import Path
import time

import cv2 as cv
import numpy as np

## COMPUTER VISION FUNCTIONS (swap them as needed for testing)

def calc_cone_polygon(img_color: np.array) -> np.array:
	"""
	Calculates polygon around the traffic cone. 
	"""
	#TODO: Histogram matching.
	# Boundaries
	lower, upper = (
		np.array([0, 0, 200], dtype=np.uint8), 
		np.array([121, 151, 255], dtype=np.uint8)
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
	polygon = np.zeros_like(img_edges)
	cv.drawContours(
		polygon, sorted_cnts, -1, (255, 255, 255), cv.FILLED
	)        
	return polygon



## FILE HANDLER FUNCTIONS (do not modify)

def read_npy_dir(root_path: str) -> Tuple[list, list]:
	"""
	Reads .npy files with the given pattern, and returns
	them in a sorted pair of lists (color and depth image).
	"""
	img_color = sorted(
		Path(root_path).glob('frame_color_*.npy'), key=lambda path: int(path.stem.rsplit("_", 1)[1])
	)
	img_depth = sorted(
		Path(root_path).glob('frame_depth_*.npy', ), key=lambda path: int(path.stem.rsplit("_", 1)[1])
	)
	return img_color, img_depth


def read_mp4_file(file_name: str) -> str:
	"""
	Loads .mp4 files.
	"""
	return Path(__file__).parent / 'test_videos' / f'{file_name}'


def read_image_file(img_path: str) -> None:
	"""
	Loads a single image (with custom processing).
	"""
	img_color = cv.imread(img_path)
	print('[INFO] Image Loaded! (dtype: %s)' % img_color.dtype)

	# Custom functions here.
	frame_processed = calc_cone_polygon(img_color)

	cv.imshow("Color Image", img_color)
	cv.imshow("Processed Image", frame_processed)
	cv.waitKey(0)

	print("[INFO] Image closed.")


def play_npy_dir(root_path: str, sleep_time: float=0.01) -> None:
	"""
	Loops trough the .npy frames (color and depth) and plays them
	as a video sequence (with custom processing).
	"""
	(img_color_list, img_depth_list) = read_npy_dir(root_path)
	for i in range(0, len(img_color_list)):
		img_color = np.load(img_color_list[i])
		img_depth = np.load(img_depth_list[i])


		# Custom functions here.
		frame_processed = calc_cone_polygon(img_color)

		
		cv.imshow("Color Image", img_color)
		cv.imshow("Processed Image", frame_processed)
		if cv.waitKey(1) == ord('q'): break
		time.sleep(sleep_time)

	print("[INFO] Video sequence end.")


def play_mp4_file(file_name: str, is_resize: bool=True) -> None:
	"""
	Plays .mp4 video file (with custom processing).
	"""
	cap = cv.VideoCapture(str(read_mp4_file(file_name)))
	while True:
		ret, img_color = cap.read()
		if ret:
			if is_resize:
				img_color = cv.resize(
					img_color, (int(img_color.shape[1] / 2.0), int(img_color.shape[0] / 2.0))
				)
			
			
			# Custom functions here.
			frame_processed = calc_cone_polygon(img_color)

			
			cv.imshow("Color Image", img_color)
			cv.imshow("Processed Image", frame_processed)
		else:
			# loops the video when there is no return value
			cap.set(cv.CAP_PROP_POS_FRAMES, 0)

		if cv.waitKey(1) & 0xFF == ord('q'): break
	cap.release()
	print("[INFO] Video sequence end.")
	cv.destroyAllWindows()

## MAIN PARTS

def main(arg: str) -> None:

	# image
	read_image_file(arg)
	
	# .npy
	#play_npy_dir(arg)

	# .mp4
	#play_mp4_file(arg)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"-d",
		"--dir",
		default='./meresek/rec5/', 
		help = "Path to .npy directory."
	)
	ap.add_argument(
		"-v",
		"--video",
		default='20210323_113620.mp4', 
		help = "Path to video file."
	)
	ap.add_argument(
		"-i",
		"--image",
		default='src/test_imgs/boja.png', 
		help = "Path to image."
	)
	args = vars(ap.parse_args())
	# change argument type if needed + switch function in main()
	main(args["image"])
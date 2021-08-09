from typing import Tuple
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


# Custom test function
def custom_test(frame: np.array) -> np.array:
	"""
	Test function for file loaders.
	"""
	frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	return frame


def read_npy_file(root_path: str, idx: int) -> Tuple[np.array, np.array]:
	"""
	Loads a pair of .npy files (color and depth image).
	"""
	img_color = np.load(Path(root_path) / "frame_color_" + str(idx) + ".npy")
	img_depth = np.load(Path(root_path) / "frame_depth_" + str(idx) + ".npy")
	return (img_color, img_depth)


def read_mp4_file(file_name: str, ext: str='mp4') -> str:
	"""
	Loads .mp4 files.
	"""
	return Path(__file__).parent / 'test_videos' / f'{file_name}.{ext}'


def play_npy_file():
	"""
	"""
	pass


def play_mp4_file(file_name: str, is_resize: bool=True):
	"""
	"""
	cap = cv.VideoCapture(str(read_mp4_file(file_name)))
	while True:
		ret, frame = cap.read()
		if ret:
			if is_resize:
				frame = cv.resize(
					frame, (int(frame.shape[1] / 2.0), int(frame.shape[0] / 2.0))
				)
			
			
			# Custom functions here.
			frame_processed = custom_test(frame)

			
			cv.imshow("Raw Image", frame)
			cv.imshow("Processed Image", frame_processed)
		else:
			# loops the video when there is no return value
			cap.set(cv.CAP_PROP_POS_FRAMES, 0)

		if cv.waitKey(1) & 0xFF == ord('q'): break
	cap.release()
	print("[INFO] Video sequence end.")
	cv.destroyAllWindows()


def main() -> None:
	play_mp4_file('20210323_113620')


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"-i",
		"--image",
		default='src/test_imgs/boja.png', 
		help = "Path to the image."
	)
	args = vars(ap.parse_args())
	main()
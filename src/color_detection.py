import sys
import numpy as np
import argparse
import cv2 as cv
import imutils
from imutils.perspective import four_point_transform
from skimage import exposure
import matplotlib.pyplot as plt


def find_color_card(image: np.array) -> None:
	"""
	Loads the ArUCo dictionary, grab the ArUCo parameters, and
	detect the markers in the input image.
	Params:
		image: contains the color card
	"""
	arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
	arucoParams = cv.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv.aruco.detectMarkers(image,
		arucoDict, parameters=arucoParams)

	# try to extract the coordinates of the color correction card
	try:
		# otherwise, we've found the four ArUco markers, so we can
		# continue by flattening the ArUco IDs list
		ids = ids.flatten()

		# extract the top-left marker
		i = np.squeeze(np.where(ids == 923))
		topLeft = np.squeeze(corners[i])[0]

		# extract the top-right marker
		i = np.squeeze(np.where(ids == 1001))
		topRight = np.squeeze(corners[i])[1]

		# extract the bottom-right marker
		i = np.squeeze(np.where(ids == 241))
		bottomRight = np.squeeze(corners[i])[2]

		# extract the bottom-left marker
		i = np.squeeze(np.where(ids == 1007))
		bottomLeft = np.squeeze(corners[i])[3]

	# we could not find color correction card, so gracefully return
	except:
		return None

	# build our list of reference points and apply a perspective
	# transform to obtain a top-down, birds-eye-view of the color
	# matching card
	cardCoords = np.array([topLeft, topRight,
		bottomRight, bottomLeft])
	card = four_point_transform(image, cardCoords)

	# return the color matching card to the calling function
	return card

def create_poly_updated(image):

	# first array: x >= , second array: x <= (B, G, R)
	boundaries = ([0, 0, 200], [121, 151, 255])

	lower, upper = boundaries
	lower = np.array(lower, dtype=np.uint8)
	upper = np.array(upper, dtype=np.uint8)

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(image, lower, upper)
	output = cv.bitwise_and(image, image, mask= mask)
	
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
	closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)
	blurred = cv.medianBlur(closing, 5)
	img_edges = cv.Canny(blurred, 30, 160)
	cnts, hierarcy = cv.findContours(img_edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	contours = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
	img_cnts = np.zeros_like(image, dtype=np.uint8)
	cv.drawContours(img_cnts, contours, -1, (255, 255, 255), cv.FILLED)


def main(img_path: str) -> None:
	"""
	Main function.
	"""
	image = cv.imread(img_path)
	print('[INFO] Image Loaded! (dtype: %s)' % image.dtype)

	# first array: x >= , second array: x <= (B, G, R)
	boundaries = ([0, 0, 200], [121, 151, 255])
	
	lower, upper = boundaries
	lower = np.array(lower, dtype=np.uint8)
	upper = np.array(upper, dtype=np.uint8)

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(image, lower, upper)
	output = cv.bitwise_and(image, image, mask= mask)
	

	kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
	closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)
	blurred = cv.medianBlur(closing, 5)

	img_edges = cv.Canny(blurred, 30, 160)

	# cnts have a length of 2, it contains the contours (cnts[0]), and the 
	# hierarchy (cnts[1])
	# the hierarchy have the shape (1, len(cnts[0]), 4)
	cnts, hierarcy = cv.findContours(img_edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	
	# approx_contours = []
	# for c in cnts:
	# 	peri = cv.arcLength(c, True)
	# 	approx = cv.approxPolyDP(c, 0.015 * peri, True)
	# 	approx_contours.append(approx)

	contours = sorted(cnts, key=cv.contourArea, reverse=True)[:1]

	polygon = np.zeros_like(image, dtype=np.uint8)
	cv.fillPoly(polygon, contours, (255, 255, 255))

	# img_cnts = np.zeros_like(image, dtype=np.uint8)
	# cv.drawContours(img_cnts, contours, -1, (255,255,255), cv.FILLED)

		# show the images
	cv.imshow("images", np.hstack([polygon, image, output]))
	cv.waitKey(0)
	# cv.imshow("blurred", image)
	# cv.waitKey(0)


def working_main(img_path: str) -> None:
	"""
	#https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
	"""
	image = cv.imread(img_path)
	print('[INFO] Image Loaded! (dtype: %s)' % image.dtype)

	# first array: x >= , second array: x <= (B, G, R)
	boundaries = ([0, 0, 200], [121, 151, 255])
	
	lower, upper = boundaries
	lower = np.array(lower, dtype=np.uint8)
	upper = np.array(upper, dtype=np.uint8)

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(image, lower, upper)
	output = cv.bitwise_and(image, image, mask= mask)
	
	# show the images
	cv.imshow("images", np.hstack([image, output]))
	cv.waitKey(0)

	# postprocessing operations
	kernel_sizes = [(3, 3), (5, 5), (7, 7)]
	size = kernel_sizes[1]
	kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
	closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)

	#TODO: try bilateral blur - reduces noise while preserving edges
	blurred = cv.medianBlur(closing, 5)
	# edge detection
	img_edges = cv.Canny(blurred, 30, 160)
	# find contours (segments?)
	cnts = cv.findContours(img_edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv.contourArea, reverse=True)

	img_contours = np.zeros_like(img_edges)
	cv.drawContours(img_contours, cnts, -1, (255,255,255), 3)
	polygon = np.zeros_like(img_edges)
	cv.fillPoly(polygon, pts =[cnts[0]], color=(255,255,255))

	#! Think about how one could use these to filter drivable area
	#TODO: create ROS2 node (with proper structure)
	#TODO: create a testing file for the node to implement new features
	#TODO: how to calculate distance from area (like car bounding boxes)
	#TODO: fix contour area switching problem
	#TODO: Put convex around the contour (research contour, convex hull)
	#TODO: histogram matching

	# show the images
	cv.imshow("Edge Detection: {}".format(
		9), polygon
	)
	cv.waitKey(0)



def color_detection(img_path: str) -> None:
	"""
	#https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
	"""
	image = cv.imread(img_path)
	print('[INFO] Image Loaded! (dtype: %s)' % image.dtype)

	# first array: x >= , second array: x <= (B, G, R)
	boundaries = ([0, 0, 200], [121, 151, 255])
	
	lower, upper = boundaries
	lower = np.array(lower, dtype=np.uint8)
	upper = np.array(upper, dtype=np.uint8)

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(image, lower, upper)
	output = cv.bitwise_and(image, image, mask= mask)

	# Alternative Solutions
	# Option 1: HSV or L*a*b* color spaces
	# Option 2: Color correction card
	# https://www.pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

	# show the images
	cv.imshow("images", np.hstack([image, output]))
	cv.waitKey(0)


def color_matching(ref_path: str, image_path: str) -> None:
	"""
	Color matching function.
	"""
	print("[INFO] loading images...")
	ref = cv.imread(ref_path)
	image = cv.imread(image_path)

	ref = imutils.resize(ref, width=600)
	image = imutils.resize(image, width=600)

	print("[INFO] finding color matching cards...")
	refCard = find_color_card(ref)
	imageCard = find_color_card(image)
	if refCard is None or imageCard is None:
		print("[INFO] could not find color matching card in both images")
		sys.exit(0)

	cv.imshow("Reference", refCard)
	cv.imshow("Input", imageCard)
	cv.waitKey(0)

	print("[INFO] matching images...")
	imageCard = exposure.match_histograms(
		imageCard, refCard, multichannel=True
	)
	# show our input color matching card after histogram matching
	cv.imshow("Input Color Card After Matching", imageCard)
	cv.waitKey(0)


def histogram_matching(ref_path: str, image_path: str) -> None:
	"""
	Histogram matching function.
	"""
	print("[INFO] loading source and reference images...")
	src = cv.imread(image_path)
	ref = cv.imread(ref_path)

	print("[INFO] performing histogram matching...")
	# determine if we are performing multichannel histogram matching
	multi = True if src.shape[-1] > 1 else False
	matched = exposure.match_histograms(src, ref, multichannel=multi)

	cv.imshow("Source", src)
	cv.imshow("Reference", ref)
	cv.imshow("Matched", matched)
	cv.waitKey(0)

	# construct a figure to display the histogram plots for each channel
	# before and after histogram matching was applied
	(fig, axs) =  plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

	# loop over our source image, reference image, and output matched
	# image
	for (i, image) in enumerate((src, ref, matched)):
		# convert the image from BGR to RGB channel ordering
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		# loop over the names of the channels in RGB order
		for (j, color) in enumerate(("red", "green", "blue")):
			# compute a histogram for the current channel and plot it
			(hist, bins) = exposure.histogram(image[..., j],
				source_range="dtype")
			axs[j, i].plot(bins, hist / hist.max())

			# compute the cumulative distribution function for the
			# current channel and plot it
			(cdf, bins) = exposure.cumulative_distribution(image[..., j])
			axs[j, i].plot(bins, cdf)

			# set the y-axis label of the current plot to be the name
			# of the current color channel
			axs[j, 0].set_ylabel(color)

	# set the axes titles
	axs[0, 0].set_title("Source")
	axs[0, 1].set_title("Reference")
	axs[0, 2].set_title("Matched")

	# display the output plots
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"-i",
		"--image",
		default='/home/norbert/Pictures/traffic_cone.jpeg', 
		help = "path to the image (to apply color correction to)"
	)
	ap.add_argument(
		"-r", 
		"--reference",
		required=False,
		help="path to the input reference image"
	)
	args = vars(ap.parse_args())
	main(args["image"])
	#color_matching(args["reference"], args["image"])
	#histogram_matching(args["reference"], args["image"])
		

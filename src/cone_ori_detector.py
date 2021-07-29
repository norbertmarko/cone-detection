#!/usr/bin/env python

from timeit import default_timer as timer

import cv2
import numpy as np
import pyrealsense2 as rs

import math

import sys

from random import randint
import time


class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # if vertical
                if 45 < orientation < 135:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all

################################################################x


#'''
pipeline = rs.pipeline()

config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)
#'''

num = 0



try:
    while (True):
        #'''
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #'''

        #path = "/home/hj/cone_detection/rec3"
        #depth_image = np.load(path + "/frame_depth_" + str(num) + ".npy")
        #color_image = np.load(path + "/frame_color_" + str(num) + ".npy")
        #num += 1
        

        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), 9)
        #color_image = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
        #color_image = cv2.resize(color_image, None, fx=0.5, fy=0.5)


        frame_HSV = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        frame_thresholded = cv2.inRange(frame_HSV, (0, 70, 171), (60, 255, 255))
        kernel = np.ones((5, 5))
        img_thresh_opened = cv2.morphologyEx(frame_thresholded, cv2.MORPH_OPEN, kernel)
        img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
        img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

        img_edges = cv2.dilate(img_edges, kernel, iterations = 5)
        
        contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        masked_frame = color_image.copy()
        
        if len(contours) > 0:
            hulled = cv2.convexHull(contours[0])

            img_hull = color_image.copy()
            cv2.drawContours(img_hull, [hulled], 0, (0, 255, 0), -1)

            masked_frame = ((color_image.astype(float) + img_hull.astype(float)) / 2).astype(np.uint8)        
        
        #cv2.imshow("masked_frame", masked_frame.astype(np.uint8))

        old_lines = color_image.copy()
        new_lines = color_image.copy()

        if len(contours) > 0:
            img_hull = np.zeros(color_image.shape).astype(np.uint8)
            cv2.drawContours(img_hull, [hulled], 0, (0, 255, 0), 2)
            img_hull = cv2.cvtColor(img_hull, cv2.COLOR_BGR2GRAY)
            lines = cv2.HoughLinesP(img_hull, 1, np.pi/180, 100)

            
            if not(lines is None):
                #merged_lines = merge_lines(lines)
                #if lines.shape[0] > 10:
                hb = HoughBundler()
                merged_lines = hb.process_lines(lines)
                for i in range(lines.shape[0]):
                    cv2.line(old_lines, (lines[i, 0, 0], lines[i, 0, 1]), (lines[i, 0, 2], lines[i, 0, 3]), (randint(0, 255), randint(0, 255), randint(0, 255)), 10)
                #print(len(merged_lines))
                max_ori = 0.0
                max_dst = 0
                max_idx = 0
                if len(merged_lines) > 0:
                    for i in range(len(merged_lines)):
                        orientation = math.atan2(abs((merged_lines[i][0][0] - merged_lines[i][1][0])), abs((merged_lines[i][0][1] - merged_lines[i][1][1])))
                        orientation = math.degrees(orientation)
                        max_point = max(merged_lines[i][0][1], merged_lines[i][1][1])
                        #if orientation > max_ori:
                        if max_point > max_dst:
                            max_ori = orientation
                            max_idx = i
                            max_dst = max_point
                    cv2.line(new_lines, (merged_lines[max_idx][0][0], merged_lines[max_idx][0][1]), (merged_lines[max_idx][1][0],merged_lines[max_idx][1][1]), (randint(0, 255), randint(0, 255), randint(0, 255)), 10)
                    new_lines = cv2.putText(new_lines, str(int(max_ori)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        
        cv2.imshow("all lines", old_lines.astype(np.uint8))
        cv2.imshow("filtered lines", new_lines.astype(np.uint8))


        #cv2.imshow("depth_image", depth_colormap)
        
        #cv2.imshow("frame", depth_image_3d.astype(np.uint8))
        if cv2.waitKey(1) == ord('q'):
            break

except:
    print("error")
finally:
    print("finally")
    pipeline.stop()
    cv2.destroyAllWindows() 



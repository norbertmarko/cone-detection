import time
import cv2
import numpy as np
from os.path import join
from pathlib import Path


extensions = ['mp4', 'npy']
sub_dir_name = 'rec0'
file_name = '20210326_120239'
ext = extensions[1]

path_mp4 = Path(__file__).parent.absolute() / 'test_videos' / f'{file_name}.{ext}'
path_npy = Path(__file__).parent.parent.absolute() / 'meresek' / sub_dir_name 


max_value = 255
max_value_H = 360//2

low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

def read_npy_file(path, idx):
    img_depth = np.load(join(path, "frame_depth_" + str(idx) + ".npy"))
    img_color = np.load(join(path, "frame_color_" + str(idx) + ".npy"))
    return (img_depth, img_color)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

# create and configure the windows
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)

cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

if ext == 'mp4':

    cap = cv2.VideoCapture(str(path_mp4))

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (int(frame.shape[1] / 2.0), int(frame.shape[0] / 2.0)))
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

            cv2.imshow(window_capture_name, frame)
            cv2.imshow(window_detection_name, frame_threshold)
        else:
            # loops the video when there is no return value
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    print("[INFO] Video sequence end.")
    cv2.destroyAllWindows()

elif ext == 'npy':

    datatype = np.uint8
    sleep_time = 0.01
    idx = 0

    while True:
        try: 
            _ , frame = read_npy_file(path_npy, idx)
            idx += 1
            frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

            cv2.imshow(window_capture_name, frame)
            cv2.imshow(window_detection_name, frame_threshold)

            # required for proper display
            if cv2.waitKey(1) == ord('q'): break
            time.sleep(sleep_time)   
        except Exception as e:
            #print(f'Problem: {e}')
            idx = 0
    print("[INFO] Video sequence end.")
    cv2.destroyAllWindows()
else:
    raise TypeError('Not a proper extension.')



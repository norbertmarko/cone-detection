import numpy as np
import cv2


cap = cv2.VideoCapture("test_videos/20210326_120239.mp4")

counter = 0
w = 672
h = 376

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter('cone_test.avi', fourcc, 30.0, (w * 2, h), True)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret is True:
        counter += 1
        print("Valid frame: {}".format(counter))
        pass
    else:
        print("Video finished!")
        break

    frame = cv2.resize(frame, (int(frame.shape[1] / 2.0), int(frame.shape[0] / 2.0)))

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # (low_H, low_S, low_V), (high_H, high_S, high_V)
    frame_thresholded = cv2.inRange(frame_HSV, (0, 70, 171), (60, 255, 255))

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(frame_thresholded, cv2.MORPH_OPEN, kernel)
    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
    
    # edge and contour
    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)

    # approx. contours
    approx_contours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    # video ckpt
    img_approx_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

    # convex hulls
    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))

    img_all_convex_hulls = np.zeros_like(img_edges)
    cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)

    cv2.drawContours(frame, convex_hulls_3to10, -1, (255 ,0, 0), 3)

    # # create 3 channel image from 1
    # img_convex_hulls_3to10 = cv2.merge(
    #     (img_convex_hulls_3to10,img_convex_hulls_3to10,img_convex_hulls_3to10)
    # )


    output = np.zeros((h * 1, w * 2, 3), dtype="uint8")
    output[0:h, 0:w] = np.uint8(cv2.resize(frame_HSV, (672, 376)))
    output[0:h, w:w * 2] = np.uint8(cv2.resize(frame, (672, 376)))

    video.write(output)

video.release()

"""
    cv2.imshow("frame", img_convex_hulls_3to10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



img_array = []
seg_array = []

w = 672
h = 376
test_set_length = 1960 #841 #797 #554 #892 #1960

img_output_path = r'/mnt/10b7da77-e382-498c-96c3-114c38319e0a/Norbi/datasets/sze/camera_zala'
img_library_path = r'/mnt/10b7da77-e382-498c-96c3-114c38319e0a/Norbi/misc/camera_test/mobilenet-master/training/video_inference/output_images'

def getint(name):
    _, num = name.split('_')
    num, _ = num.split('.')
    return int(num)

img_list = []
seg_list = []

img_file_list = os.listdir(img_library_path)
seg_file_list = os.listdir(img_output_path)

img_file_list = sorted(img_file_list, key=getint)
seg_file_list = sorted(seg_file_list, key=getint)

for i in range(test_set_length):
    img_list.append( os.path.join(img_library_path, img_file_list[i]) )
    seg_list.append( os.path.join(img_output_path, seg_file_list[i]) )

#print(img_list[:2], seg_list[:2])


for i in range(len(os.listdir(img_library_path))):
    
    img = cv2.imread(img_list[i])
    seg = cv2.imread(seg_list[i])

    img_array.append(img)
    seg_array.append(seg)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter('csorna_lane_comparation.avi', fourcc, 20.0, (w * 2, h), True)

for i in range(len(img_array)):
    print(i)
    output = np.zeros((h * 1, w * 2, 3), dtype="uint8")
    output[0:h, 0:w] = np.uint8(cv2.resize(img_array[i], (672, 376)))
    output[0:h, w:w * 2] = np.uint8(seg_array[i])

    video.write(output)

video.release()
"""
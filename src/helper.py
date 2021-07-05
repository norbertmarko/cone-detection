import cv2

def convex_hull_pointing_up(ch):
    '''Determines if the path is pointing up.
    If so, this is a cone.'''
        
    # contour points above center and below
    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv2.boundingRect(ch) # coordinates of the upper-left corner of the enclosing rectangle, width and height
    aspect_ratio = w / h # ratio of rectangle width to height

    # if the rectangle is narrow, continue defining. If not, then the circuit does not fit
    if aspect_ratio < 0.8:
        # each point of the contour is classified as lying above or below the center
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center: # if the y-coordinate of the point is above the center, then add this point to the list of points above the center
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        # define the x coordinates of the extreme points below the center
        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        # check if the top points of the path are outside the "base". If yes, then the circuit does not fit
        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False    
    return True

# related code draft

# cones = []
# bounding_rects = []
# for ch in convex_hulls_3to10:
#     if convex_hull_pointing_up(ch):
#         cones.append(ch)
#         rect = cv2.boundingRect(ch)
#         bounding_rects.append(rect)

# img_cones = np.zeros_like(img_edges)
# cv2.drawContours(img_cones, cones, -1, (255,255,255), 2)
# cv2.drawContours(img_cones, bounding_boxes, -1, (1,255,1), 2)
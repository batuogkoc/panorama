import cv2
import numpy as np
import math as m
from copy import deepcopy
def find_distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return m.sqrt((x1-x2)**2 + (y1-y2)**2)

def find_slope(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return m.atan2(y2-y1, x2-x1)

def find_handedness(p1, p2, p3):
	if np.cross(p3-p2, p2-p1)>0:
		return True
	else:
		return False

def find_angle(p_prev, p_curr, p_next):
	slope_prev = find_slope(p_prev, p_curr)
	slope_next = find_slope(p_curr, p_next)
	angle = (slope_next-slope_prev)*(180/m.pi)%360.0
	return angle

def find_curvature(p_prev, p_curr, p_next):
	angle = find_angle(p_prev, p_curr, p_next)
	dist = find_distance(p_curr, p_next)
	return angle/dist

def find_corners(img):
    frame = img
    kernel = np.ones((5,5), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,9,50,50)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = deepcopy(frame)
    cv2.drawContours(draw, contours, -1, (0, 255, 255), 3)
    contours = tuple(x for x in contours if cv2.contourArea(x) > 1000)

    right_handed_corners = []
    left_handed_corners = []
    other_corners = []

    if len(contours) >0:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        for contour in contours:
            epsilon = 0.015*cv2.arcLength(contour, False)
            
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for i in range(len(approx)):
                p_prev = approx[(i-1)%len(approx)][0]
                p_curr = approx[(i)%len(approx)][0]
                p_next = approx[(i+1)%len(approx)][0]

                curvature = find_curvature(p_prev, p_curr, p_next)
                angle = find_angle(p_prev, p_curr, p_next)
                epsilon = 20
                curvature_min = 1
                if curvature>curvature_min or abs(angle-90)<epsilon:
                    if find_handedness(p_prev, p_curr, p_next):
                        left_handed_corners.append(p_curr)
                    else:
                        right_handed_corners.append(p_curr)
                else:
                    other_corners.append(p_curr)
    return right_handed_corners, left_handed_corners, other_corners 

if __name__ == "__main__":
    cap = cv2.VideoCapture('output.avi')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter("corners_1.mp4", fourcc, 15, (1126, 720))

    assert cap.isOpened(), "Error opening video stream or file"

    try:
        while(cap.isOpened()):
            ret, frame = cap.read()

            h, w, d = np.shape(frame)
            frame_masked = deepcopy(frame)
            blank = np.zeros((h,w), np.uint8)
            trapezoid = np.array([[400, 570],
                                [730, 570],
                                [730, 500],
                                [600, 410],
                                [530, 410],
                                [400, 500]], np.int32)
            trapezoid = trapezoid.reshape((-1, 1, 2))
            cv2.fillPoly(blank, [trapezoid], (255, 255, 255))
            frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=cv2.bitwise_not(blank))

            if ret == True:
                right_handed_corners, _, _ = find_corners(frame_masked)
                for corner in right_handed_corners:
                    cv2.circle(frame, tuple(corner), 5, (0,0,255),thickness = 3) 
                                           
                cv2.imshow('Frame',frame)
                writer.write(frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else: 
                break
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
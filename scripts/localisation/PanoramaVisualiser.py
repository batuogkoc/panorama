# some_file.py
import sys
import pickle
import cv2
import numpy as np
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/mapping')
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/python_tests/map_annotator')
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/python-utils')

from copy import deepcopy
from utils import *
import annotator

def mouse(event, x, y, flags, param):
    global panorama
    global resizable_image
    point = (x,y)
    x, y=resizable_image.out_to_in_point(point)
    coords = panorama.pixel_to_meters(np.array([[x],[y]]))
    print(f"x: {coords[0][0]:4.1f} y: {coords[1][0]:4.1f}")

if __name__=="__main__":
    with open("map.panorama", "rb") as f:
        panorama = pickle.load(f)
    frame = cv2.dilate(deepcopy(panorama.get_output_img()), (3,3), iterations=2)
    cv2.imwrite("map.png", frame)
    resizable_image = annotator.ResizableImage(0.5)
    cv2.namedWindow("win")
    cv2.setMouseCallback("win", mouse)
    while True:
        cv2.imshow("win", resizable_image.resize(frame))
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
#usr/bin/env python

from CornerDetector import find_corners
import rospy
import Panorama
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import std_msgs.msg
import geometry_msgs
import tf2_ros
import tf.transformations as ts
import tf
import numpy as np
import cv_bridge
import cv2
import time
import math as m
from copy import deepcopy
from utils import *
from matplotlib import pyplot as plt
import pickle

tf_buffer = tf2_ros.Buffer()
tf_listener = None
bridge = cv_bridge.CvBridge()

source = "odom"
camera_transform = None

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter("output.avi", fourcc, 15, (1920, 1080))
first_time = True

main_time = 0
right_time = 0
left_time = 0

rate_control_publisher = None

def camera_callback(data):
    frame_id = "camera"
    global tf_buffer
    global panorama
    global bridge
    global source
    global camera_transform
    # try:
    camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
    img = bridge.compressed_imgmsg_to_cv2(data)
    panorama.cameras[frame_id].add_image_stamped(data.header.stamp, img)
    # except Exception as e:
    #     rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))
    #     return

    main()


def main():
    times = Times()
    global camera_transform
    try:
        camera_htm = geometry_msgs_TransformStamped_to_htm(camera_transform)
    except AttributeError:
        rospy.loginfo("waiting for tf")
        return
    print(camera_htm)
    panorama.cameras["camera"].add_htm(camera_htm, camera_transform.header.stamp)

    times.add("htm")
    # panorama.clear_img()
    panorama.project_camera("camera", extrapolate_htm=True)
    frame = cv2.dilate(deepcopy(panorama.get_output_img()), (3,3))
    times.add("project")
    
    resized_img = imshow_r("a", frame, (1600, 900))
    cv2.waitKey(1)
    print(panorama.image_append.map_corner_coords)
    # print(panorama.pixel_to_meters(np.zeros((2,1))))
    # out.write(resized_img)
    times.add("imshow")
    print(times)

def node():
    global panorama
    global tf_listener
    
    rospy.init_node("panorama", anonymous=True)
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("camera/image/compressed", CompressedImage, camera_callback)
    
    panorama = Panorama.Panorama(0.2)

    camera = Panorama.Camera(1920, 1080, m.pi*60/180, 30, 30, "camera")
    camera.set_orthogonal_roi(0, camera.width, 0, camera.height)
    panorama.add_camera(camera)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except Exception as e:
        rospy.logerr(e)
    finally:
        rospy.loginfo("exiting")
        # out.release()
        # with open("map.panorama", "wb") as f:
        #     pickle.dump(panorama, f)
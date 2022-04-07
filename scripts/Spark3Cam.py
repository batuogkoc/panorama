#usr/bin/env python

from CornerDetector import find_corners
import rospy
import Panorama
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
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

tf_buffer = tf2_ros.Buffer()
tf_listener = None
bridge = cv_bridge.CvBridge()

source = "odom"
main_camera_transform = None
left_camera_transform = None
right_camera_transform = None
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter("output.avi", fourcc, 15, (1920, 1080))
first_time = True

main_time = 0
right_time = 0
left_time = 0

def left_camera_callback(data):
    frame_id = "left_camera"
    t = rospy.Time.now()
    global tf_buffer
    global panorama
    global bridge
    global source
    global left_camera_transform
    try:
        left_camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
        img = bridge.compressed_imgmsg_to_cv2(data)
        panorama.cameras[frame_id].add_image_stamped(t, img)
    except Exception as e:
        rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))

def right_camera_callback(data):
    frame_id = "right_camera"
    t = rospy.Time.now()
    global tf_buffer
    global panorama
    global bridge
    global source
    global right_camera_transform
    try:
        right_camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
        img = bridge.compressed_imgmsg_to_cv2(data)
        panorama.cameras[frame_id].add_image_stamped(t, img)
    except Exception as e:
        rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))


def main_camera_callback(data):
    frame_id = "main_camera"
    t = rospy.Time.now()
    times = Times()
    global tf_buffer
    global panorama
    global bridge
    global source
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform

    try:
        main_camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
        img = bridge.compressed_imgmsg_to_cv2(data)
        panorama.cameras[frame_id].add_image_stamped(t, img)
    except Exception as e:
        rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))


    left_camera_htm = geometry_msgs_TransformStamped_to_htm(left_camera_transform)
    right_camera_htm = geometry_msgs_TransformStamped_to_htm(right_camera_transform)
    main_camera_htm = geometry_msgs_TransformStamped_to_htm(main_camera_transform)
    try:
        map_tranform = tf_buffer.lookup_transform(source, "map", rospy.Time(0), rospy.Duration(1.0))
        map_tranform_htm = geometry_msgs_TransformStamped_to_htm(map_tranform)
    except:
        rospy.logerr("couldnt get map transform")
        return
    
    x_len = m.sqrt(map_tranform_htm[0][0]**2 + map_tranform_htm[0][1]**2)
    y_len = m.sqrt(map_tranform_htm[1][0]**2 + map_tranform_htm[1][1]**2)

    map_tranform_htm_only_z_rot = np.copy(map_tranform_htm)
    map_tranform_htm_only_z_rot[:,0] = map_tranform_htm_only_z_rot[:,0]/x_len
    map_tranform_htm_only_z_rot[:,1] = map_tranform_htm_only_z_rot[:,1]/y_len
    map_tranform_htm_only_z_rot[:,2] = np.array([[0,0,1,0]])
    map_tranform_htm_only_z_rot[2,0:2] = np.array([0,0])

    map_tranform_htm_only_z_rot_inv = ts.inverse_matrix(map_tranform_htm_only_z_rot)

    main_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, main_camera_htm)
    right_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, right_camera_htm)
    left_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, left_camera_htm)

    panorama.cameras["left_camera"].set_htm(left_local_htm, left_camera_transform.header.stamp)
    panorama.cameras["right_camera"].set_htm(right_local_htm, right_camera_transform.header.stamp)
    panorama.cameras["main_camera"].set_htm(main_local_htm, main_camera_transform.header.stamp)
    times.add("htm")
    panorama.clear_img()
    panorama.project_all_cameras()
    frame = deepcopy(panorama.get_output_img())
    times.add("project")
    # h, w, d = np.shape(frame)
    # frame_masked = deepcopy(frame)
    # blank = np.zeros((h,w), np.uint8)
    # trapezoid = np.array([[400, 570],
    #                     [730, 570],
    #                     [730, 500],
    #                     [600, 410],
    #                     [530, 410],
    #                     [400, 500]], np.int32)
    # trapezoid = trapezoid.reshape((-1, 1, 2))
    # cv2.fillPoly(blank, [trapezoid], (255, 255, 255))
    # frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=cv2.bitwise_not(blank))
    # right_handed_corners, left_handed_corners, other_corners = find_corners(frame_masked.astype(np.uint8))

    # for corner in right_handed_corners:
    #     cv2.circle(frame, tuple(corner), 5, (0,0,255),thickness = 3) 
    # for corner in left_handed_corners:
    #     cv2.circle(frame, tuple(corner), 5, (0,255,0),thickness = 3) 
    # for corner in other_corners:
    #     cv2.circle(frame, tuple(corner), 5, (255,0,0),thickness = 3) 
    # cv2.imwrite("img.jpg", frame.astype(np.float32))
    resized_img = imshow_r("a", frame, (1280, 720))
    cv2.waitKey(1)
    global first_time

    # out.write(resized_img)
    times.add("imshow")
    print(times)
    # print(1/duration_to_sec(rospy.Time.now()-t))

def node():
    global panorama
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform
    rospy.init_node("panorama", anonymous=True)
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("main_camera/image/compressed", CompressedImage, main_camera_callback)
    rospy.Subscriber("left_camera/image/compressed", CompressedImage, left_camera_callback)
    rospy.Subscriber("right_camera/image/compressed", CompressedImage, right_camera_callback)

    panorama = Panorama.Panorama(0.03)

    get_tf = True
    while get_tf:
        try:
            main_camera_transform = tf_buffer.lookup_transform("map", "main_camera", rospy.Time(0), rospy.Duration(1.0))
            left_camera_transform = tf_buffer.lookup_transform("map", "left_camera", rospy.Time(0), rospy.Duration(1.0))
            right_camera_transform = tf_buffer.lookup_transform("map", "right_camera", rospy.Time(0), rospy.Duration(1.0))
            
            # f = (1080/2)*m.tan(m.pi*60/180)
            # l_r_fov = 2*m.atan(1920/2/f)
            # main_fov = 2*m.atan(1250/2/f)

            # pos,rot = geometry_msgs_TransformStamped_to_pos_rot(main_camera_transform)
            # panorama.add_camera("main_camera", pos, rot, 1250, 1080, main_fov)

            # pos,rot = geometry_msgs_TransformStamped_to_pos_rot(left_camera_transform)
            # panorama.add_camera("left_camera", pos, rot, 1920, 1080, l_r_fov)

            # pos,rot = geometry_msgs_TransformStamped_to_pos_rot(right_camera_transform)
            # panorama.add_camera("right_camera", pos, rot, 1920, 1080, l_r_fov)
            a = 0.66
            left_camera = Panorama.Camera(1920, 1080, m.pi*60/180, 30, "left_camera")
            left_camera.set_orthogonal_roi(0, left_camera.width, int(left_camera.height*(a)), left_camera.height)
            right_camera = Panorama.Camera(1920, 1080, m.pi*60/180, 30, "right_camera")
            right_camera.set_orthogonal_roi(0, right_camera.width, int(right_camera.height*(a)), right_camera.height)
            main_camera = Panorama.Camera(1250, 1080, m.pi*60/180, 30, "main_camera")
            main_camera.set_orthogonal_roi(0, main_camera.width, int(main_camera.height*(a)), main_camera.height)


            panorama.add_camera(left_camera)
            panorama.add_camera(right_camera)
            panorama.add_camera(main_camera)

            get_tf = False
        except Exception as e:
            rospy.logerr(e)
    rospy.spin()
        
if __name__ == "__main__":
    try:
        node()
    except Exception as e:
        rospy.logerr(e)
    finally:
        rospy.loginfo("exiting")
        out.release()
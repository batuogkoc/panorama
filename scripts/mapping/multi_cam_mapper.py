#usr/bin/env python3
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../multi_cam_mapping')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../python-utils')))
print(sys.path)
print(script_dir)

import rospy
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
from python_utils.utils import *
from multi_cam_mapping.multi_cam_project import MultiCamProject, Camera


tf_buffer = tf2_ros.Buffer()
tf_listener = None
bridge = cv_bridge.CvBridge()

source = "odom"
main_camera_transform = None
left_camera_transform = None
right_camera_transform = None
# out = cv2.VideoWriter("output.avi", fourcc, 15, (1920, 1080))
first_time = True

main_time = 0
right_time = 0
left_time = 0

rate_control_publisher = None
project = None

class MultiCamMapper:
    def __init__(self):
        self.source_frame_id = rospy.get_param(param_name)

def left_camera_callback(data):
    frame_id = "left_camera"
    t = data.header.stamp
    global tf_buffer
    global project
    global bridge
    global source
    global left_camera_transform
    try:
        left_camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
        img = bridge.compressed_imgmsg_to_cv2(data)
        project.cameras[frame_id].add_image_stamped(t, img)
    except Exception as e:
        rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))

def right_camera_callback(data):
    frame_id = "right_camera"
    t = data.header.stamp
    global tf_buffer
    global project
    global bridge
    global source
    global right_camera_transform
    try:
        right_camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
        img = bridge.compressed_imgmsg_to_cv2(data)
        project.cameras[frame_id].add_image_stamped(t, img)
    except Exception as e:
        rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))


def main_camera_callback(data):
    frame_id = "main_camera"
    t = data.header.stamp
    global tf_buffer
    global project
    global bridge
    global source
    global main_camera_transform

    try:
        main_camera_transform = tf_buffer.lookup_transform(source, frame_id, rospy.Time(0), rospy.Duration(1.0))
        frame = bridge.compressed_imgmsg_to_cv2(data)
        h, w, d = np.shape(frame)
        frame_masked = deepcopy(frame)
        blank = np.zeros((h,w), np.uint8)
        x_a = 70
        x_b = 180
        y_b = 960
        x_c = 500
        y_c = 890
        trapezoid = np.array([[x_a, 1080],
                              [1250-x_a, 1080],
                              [1250-x_b, y_b],
                              [1250-x_c, y_c],
                              [x_c, y_c],
                              [x_b, y_b]], np.int32)
        trapezoid = trapezoid.reshape((-1, 1, 2))
        cv2.fillPoly(blank, [trapezoid], (255, 255, 255))
        frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=cv2.bitwise_not(blank))
        project.cameras[frame_id].add_image_stamped(t, frame_masked)
    except Exception as e:
        rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))
    
    main()

def main():
    times = Times()
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform
    try:
        left_camera_htm = geometry_msgs_TransformStamped_to_htm(left_camera_transform)
        right_camera_htm = geometry_msgs_TransformStamped_to_htm(right_camera_transform)
        main_camera_htm = geometry_msgs_TransformStamped_to_htm(main_camera_transform)
    except AttributeError:
        rospy.loginfo("waiting for tf")
        return
    try:
        map_tranform = tf_buffer.lookup_transform(source, "map", rospy.Time(0), rospy.Duration(1.0))
        map_tranform_htm = geometry_msgs_TransformStamped_to_htm(map_tranform)
    except:
        rospy.logerr("couldnt get map transform")
        return
    
    # x_len = m.sqrt(map_tranform_htm[0][0]**2 + map_tranform_htm[0][1]**2)
    # y_len = m.sqrt(map_tranform_htm[1][0]**2 + map_tranform_htm[1][1]**2)

    # map_tranform_htm_only_z_rot = np.copy(map_tranform_htm)
    # map_tranform_htm_only_z_rot[:,0] = map_tranform_htm_only_z_rot[:,0]/x_len
    # map_tranform_htm_only_z_rot[:,1] = map_tranform_htm_only_z_rot[:,1]/y_len
    # map_tranform_htm_only_z_rot[:,2] = np.array([[0,0,1,0]])
    # map_tranform_htm_only_z_rot[2,0:2] = np.array([0,0])

    # map_tranform_htm_only_z_rot_inv = ts.inverse_matrix(map_tranform_htm_only_z_rot)

    # main_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, main_camera_htm)
    # right_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, right_camera_htm)
    # left_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, left_camera_htm)

    main_local_htm = main_camera_htm
    left_local_htm = left_camera_htm
    right_local_htm = right_camera_htm

    project.cameras["left_camera"].add_htm(left_local_htm, left_camera_transform.header.stamp)
    project.cameras["right_camera"].add_htm(right_local_htm, right_camera_transform.header.stamp)
    project.cameras["main_camera"].add_htm(main_local_htm, main_camera_transform.header.stamp)

    times.add("htm")
    # project.clear_img()
    # project.project_all_cameras(extrapolate_htm=False)
    project.project_camera("main_camera", extrapolate_htm=True)
    times.add("project")
    # frame = cv2.dilate(deepcopy(project.get_output_img()), (3,3),iterations=2)
    # # right_handed_corners, left_handed_corners, other_corners = find_corners(frame)
    # # for corner in right_handed_corners:
    # #     cv2.circle(frame, corner, 3, (255,0,0), thickness=1)
    # # times.add("detect corners")
    # resized_img = imshow_r("a", frame, (1600, 900))
    # cv2.waitKey(1)

    # out.write(resized_img)
    # times.add("imshow")
    print(times)


def node():
    global project
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform
    rospy.init_node("multi_cam_mapper", anonymous=True)
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    project = MultiCamProject(0.03)
    a = 0.58
    left_camera = Camera(1920, 1080, m.pi*60/180, 30, 30, "left_camera")
    left_camera.set_orthogonal_roi(0, left_camera.width, int(left_camera.height*(a)), left_camera.height)
    right_camera = Camera(1920, 1080, m.pi*60/180, 30, 30, "right_camera")
    right_camera.set_orthogonal_roi(0, right_camera.width, int(right_camera.height*(a)), right_camera.height)
    main_camera = Camera(1250, 1080, m.pi*60/180, 30, 30, "main_camera")
    main_camera.set_orthogonal_roi(0, main_camera.width, int(main_camera.height*(a)), main_camera.height)


    project.add_camera(left_camera)
    project.add_camera(right_camera)
    project.add_camera(main_camera)

    rospy.Subscriber("main_camera/image/compressed", CompressedImage, main_camera_callback)
    rospy.Subscriber("left_camera/image/compressed", CompressedImage, left_camera_callback)
    rospy.Subscriber("right_camera/image/compressed", CompressedImage, right_camera_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except rospy.ROSInterruptException as e:
        rospy.loginfo("exiting")
    finally:
        out_img = project.get_image()
        print(np.shape(out_img))
        # frame = cv2.dilate(deepcopy(project.get_output_img()), (3,3),iterations=2)
        # right_handed_corners, left_handed_corners, other_corners = find_corners(frame)
        # for corner in right_handed_corners:
        #     cv2.circle(frame, corner, 3, (255,0,0), thickness=1)
        # times.add("detect corners")
        resized_img = imshow_r("a", out_img, (1600, 900))
        while True:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("r"):
                cv2.imwrite("out-resized.jpg", resized_img)
                break
            elif key ==ord("s"):
                cv2.imwrite("out.jpg", out_img)
                break
        rospy.loginfo("exiting")
        # out.release()
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
from ResizableImShow import *

tf_buffer = tf2_ros.Buffer()
tf_listener = None
bridge = cv_bridge.CvBridge()

main_camera_data = None
left_camera_data = None
right_camera_data = None

source = "odom"
main_camera_transform = None
left_camera_transform = None
right_camera_transform = None
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("output.avi", fourcc, 15, (1920, 1080))
first_time = True

main_time = 0
right_time = 0
left_time = 0
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return np.array(rot_matrix)

def geometry_msgs_TransformStamped_to_pos_rot(transform):
    rotation = transform.transform.rotation
    translation = transform.transform.translation

    rot_quat = np.array([rotation.w, rotation.x, rotation.y, rotation.z])
    pos_transform = pos = np.array([[translation.x], [translation.y], [translation.z]])

    rot_matrix = quaternion_rotation_matrix(rot_quat)

    return pos_transform, rot_matrix

def htm_to_pos_rot(htm):
    pos = np.array([ts.translation_from_matrix(htm)]).T
    rot = htm[0:3,0:3]
    return pos, rot

def geometry_msgs_TransformStamped_to_htm(transform_stamped):
    trans = transform_stamped.transform.translation
    rot = transform_stamped.transform.rotation
    trans = [trans.x, trans.y, trans.z]
    rot = [rot.x, rot.y, rot.z, rot.w]
    return ts.concatenate_matrices(ts.translation_matrix(trans), ts.quaternion_matrix(rot))
    
def left_camera_callback(data):
    global left_camera_data
    global left_camera_transform
    global tf_buffer
    try:
        left_camera_transform = tf_buffer.lookup_transform(source, "left_camera", rospy.Time(0), rospy.Duration(1.0))
        left_camera_data = deepcopy(data)
    except:
        rospy.logerr("couldn't get left_camera tf")

def right_camera_callback(data):
    global right_camera_data
    global right_camera_transform
    global tf_buffer
    try:
        right_camera_transform = tf_buffer.lookup_transform(source, "right_camera", rospy.Time(0), rospy.Duration(1.0))
        right_camera_data = deepcopy(data)
    except:
        rospy.logerr("couldn't get right_camera tf")


def main_camera_callback(data):
    t = time.time()
    global main_camera_data 
    global left_camera_data
    global right_camera_data

    global source
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform

    try:
        main_camera_transform = tf_buffer.lookup_transform(source, "main_camera", rospy.Time(0), rospy.Duration(1.0))
        map_tranform = tf_buffer.lookup_transform("odom", "map", rospy.Time(0), rospy.Duration(1.0))
    except:
        rospy.logerr("couldn't get main_camera tf")
    
    left_camera_htm = geometry_msgs_TransformStamped_to_htm(left_camera_transform)
    right_camera_htm = geometry_msgs_TransformStamped_to_htm(right_camera_transform)
    main_camera_htm = geometry_msgs_TransformStamped_to_htm(main_camera_transform)
    map_tranform_htm = geometry_msgs_TransformStamped_to_htm(map_tranform)
    x_len = m.sqrt(map_tranform_htm[0][0]**2 + map_tranform_htm[0][1]**2)
    y_len = m.sqrt(map_tranform_htm[1][0]**2 + map_tranform_htm[1][1]**2)

    map_tranform_htm_only_z_rot = np.copy(map_tranform_htm)
    map_tranform_htm_only_z_rot[:,0] = map_tranform_htm_only_z_rot[:,0]/x_len
    map_tranform_htm_only_z_rot[:,1] = map_tranform_htm_only_z_rot[:,1]/y_len
    map_tranform_htm_only_z_rot[:,2] = np.array([[0,0,1,0]])
    map_tranform_htm_only_z_rot[2,0:2] = np.array([0,0])

    map_tranform_htm_only_z_rot_inv = ts.inverse_matrix(map_tranform_htm_only_z_rot)
    print(map_tranform_htm_only_z_rot)
    # main_camera_htm_inv = ts.inverse_matrix(main_camera_htm)
    # main_to_left_htm = np.matmul(main_camera_htm_inv, left_camera_htm)
    # main_to_right_htm = np.matmul(main_camera_htm_inv, right_camera_htm)

    main_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, main_camera_htm)
    right_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, right_camera_htm)
    left_local_htm = np.matmul(map_tranform_htm_only_z_rot_inv, left_camera_htm)

    pos,rot = htm_to_pos_rot(main_local_htm)
    print(pos, rot)
    panorama.update_camera_pos_rot("main_camera", pos, rot)

    pos, rot= htm_to_pos_rot(left_local_htm)
    print(pos, rot)
    panorama.update_camera_pos_rot("left_camera", pos, rot)

    pos,rot = htm_to_pos_rot(right_local_htm)
    print(pos, rot)
    panorama.update_camera_pos_rot("right_camera", pos, rot)




    # pos,rot = geometry_msgs_TransformStamped_to_pos_rot(main_camera_transform)
    # panorama.update_camera_pos_rot("main_camera", pos, rot)

    # pos,rot = geometry_msgs_TransformStamped_to_pos_rot(left_camera_transform)
    # panorama.update_camera_pos_rot("left_camera", pos, rot)

    # pos,rot = geometry_msgs_TransformStamped_to_pos_rot(right_camera_transform)
    # panorama.update_camera_pos_rot("right_camera", pos, rot)

    panorama.clear_img()
    right_camera_img = bridge.compressed_imgmsg_to_cv2(right_camera_data)
    panorama.update_camera_img("right_camera", right_camera_img)
    left_camera_img = bridge.compressed_imgmsg_to_cv2(left_camera_data)
    panorama.update_camera_img("left_camera", left_camera_img)
    main_camera_img = deepcopy(bridge.compressed_imgmsg_to_cv2(data))
    panorama.update_camera_img("main_camera", main_camera_img)

    frame = deepcopy(panorama.get_output_img())
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
    resized_img = imshow_r("a", frame, (1920, 1080))
    cv2.waitKey(1)
    global out
    global first_time

    out.write(resized_img)
    main_time = time.time()-t
    print(1/(main_time+left_time+right_time), 1/main_time)

def node():
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform

    global panorama
    rospy.init_node("panorama", anonymous=True)
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("main_camera/image/compressed", CompressedImage, main_camera_callback)
    rospy.Subscriber("left_camera/image/compressed", CompressedImage, left_camera_callback)
    rospy.Subscriber("right_camera/image/compressed", CompressedImage, right_camera_callback)



    panorama = Panorama.Panorama()

    get_tf = True
    while get_tf:
        try:
            main_camera_transform = tf_buffer.lookup_transform("map", "main_camera", rospy.Time(0), rospy.Duration(1.0))
            left_camera_transform = tf_buffer.lookup_transform("map", "left_camera", rospy.Time(0), rospy.Duration(1.0))
            right_camera_transform = tf_buffer.lookup_transform("map", "right_camera", rospy.Time(0), rospy.Duration(1.0))
            
            f = (1080/2)*m.tan(m.pi*60/180)
            l_r_fov = 2*m.atan(1920/2/f)
            main_fov = 2*m.atan(1250/2/f)

            pos,rot = geometry_msgs_TransformStamped_to_pos_rot(main_camera_transform)
            panorama.add_camera("main_camera", pos, rot, 1250, 1080, main_fov)

            pos,rot = geometry_msgs_TransformStamped_to_pos_rot(left_camera_transform)
            panorama.add_camera("left_camera", pos, rot, 1920, 1080, l_r_fov)

            pos,rot = geometry_msgs_TransformStamped_to_pos_rot(right_camera_transform)
            panorama.add_camera("right_camera", pos, rot, 1920, 1080, l_r_fov)

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
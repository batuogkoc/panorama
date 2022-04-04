#usr/bin/env python

from CornerDetector import find_corners
import rospy
import Panorama
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np
import cv_bridge
import cv2
import time
import math as m
from copy import deepcopy

# global tf_buffer
tf_buffer = tf2_ros.Buffer()
# global tf_listener
tf_listener = None
bridge = cv_bridge.CvBridge()

main_camera_data = None
left_camera_data = None
right_camera_data = None

odom_current = None
main_camera_odom = None
left_camera_odom = None
right_camera_odom = None

main_camera_transform = None
left_camera_transform = None
right_camera_transform = None
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("output.avi", fourcc, 15, (1127, 721))
first_time = True

main_time = 0
right_time = 0
left_time = 0

def odom_to_transform(odom):
    
    ret.transform.translation = odom.pose.pose.position
    ret.transform.rotation = odom.pose.pose.orientation
    ret.header = odom.header
    return ret

def main_camera_callback(data):
    global main_camera_data 
    global left_camera_data
    global right_camera_data

    global odom_current
    global main_camera_odom
    global left_camera_odom
    global right_camera_odom

    global main_camera_transform
    global left_camera_transform
    global right_camera_transform
    t = time.time()

    # main_camera_odom = deepcopy(odom_current)
    # main_camera_odom_t = odom_to_transform(main_camera_odom)
    # left_camera_odom_t = odom_to_transform(left_camera_odom)
    # right_camera_odom_t = odom_to_transform(right_camera_odom)

    # main_to_right_odom_t = tf2_ros.Trans
    right_camera_img = bridge.compressed_imgmsg_to_cv2(right_camera_data)
    panorama.update_camera_img("right_camera", right_camera_img)
    left_camera_img = bridge.compressed_imgmsg_to_cv2(left_camera_data)
    panorama.update_camera_img("left_camera", left_camera_img)
    main_camera_img = deepcopy(bridge.compressed_imgmsg_to_cv2(data))
    panorama.update_camera_img("main_camera", main_camera_img)

    frame = deepcopy(panorama.get_output_img())
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
    right_handed_corners, left_handed_corners, other_corners = find_corners(frame_masked.astype(np.uint8))

    for corner in right_handed_corners:
        cv2.circle(frame, tuple(corner), 5, (0,0,255),thickness = 3) 
    for corner in left_handed_corners:
        cv2.circle(frame, tuple(corner), 5, (0,255,0),thickness = 3) 
    for corner in other_corners:
        cv2.circle(frame, tuple(corner), 5, (255,0,0),thickness = 3) 
    cv2.imwrite("img.jpg", frame.astype(np.float32))
    cv2.imshow("a", frame)
    cv2.waitKey(1)
    global out
    global first_time

    out.write(panorama.get_output_img().astype(np.uint8))
    main_time = time.time()-t
    print(1/(main_time+left_time+right_time))
    
    
def left_camera_callback(data):
    t = time.time()
    global left_camera_data
    global right_time
    global right_camera_odom
    global odom_current
    left_camera_data = deepcopy(data)
    right_camera_odom = deepcopy(odom_current)

    left_time = time.time()-t

def right_camera_callback(data):
    t = time.time()
    global right_camera_data
    global right_time
    global right_camera_odom
    global odom_current
    right_camera_data = deepcopy(data)
    right_camera_odom = deepcopy(odom_current)
    
    right_time = time.time()-t

def odomCallback(data):
    global odom_current
    odom_current = data

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

# def tf_to_rot_pos(transform):
#     rot_quat = transform.transform.roration
#     pos_tf = transform.transform.

def node():
    global main_camera_transform
    global left_camera_transform
    global right_camera_transform

    global panorama
    rospy.init_node("panorama", anonymous=True)
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("odom", Odometry, odomCallback)
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

            pos_frame_align = np.array([[ 0.0000000,  1.0000000,  0.0000000],
                                        [-1.0000000,  0.0000000,  0.0000000],
                                        [ 0.0000000,  0.0000000,  1.0000000]])
            
            rot_frame_align = np.array([[ 0.0000000,  0.0000000, -1.0000000],
                                        [-1.0000000,  0.0000000,  0.0000000],
                                        [ 0.0000000,  1.0000000,  0.0000000]])

            translation = main_camera_transform.transform.translation
            pos = np.array([[translation.x], [translation.y], [translation.z]])
            rot = np.array([[1.0000000,  0.0000000,  0.0000000],
                            [0.0000000,  0.0000000, -1.0000000],
                            [0.0000000,  1.0000000,  0.0000000]])
            pos = np.matmul(frame_align, pos)
            panorama.add_camera("main_camera", pos, rot, 1250, 1080, main_fov)

            translation = left_camera_transform.transform.translation
            pos = np.array([[translation.x], [-translation.y], [translation.z]])
            rot = np.array([[ 0.7071068,  0.0000000,  0.7071068],
                            [ 0.7071068,  0.0000000, -0.7071068],
                            [ 0.0000000,  1.0000000,  0.0000000]])
            pos = np.matmul(frame_align, pos)

            panorama.add_camera("left_camera", pos, rot, 1920, 1080, l_r_fov)

            translation = right_camera_transform.transform.translation
            pos = np.array([[translation.x], [-translation.y], [translation.z]])
            rot = np.array([[ 0.7071068,  0.0000000, -0.7071068],
                            [-0.7071068,  0.0000000, -0.7071068],
                            [ 0.0000000,  1.0000000,  0.0000000]])
            pos = np.matmul(frame_align, pos)
            panorama.add_camera("right_camera", pos, rot, 1920, 1080, l_r_fov)


            

            # pos = np.array([[0], [1.219], [1.348]])
            # rot = np.array([[1.0000000,  0.0000000,  0.0000000],
            #                 [0.0000000,  0.0000000, -1.0000000],
            #                 [0.0000000,  1.0000000,  0.0000000]])
            # panorama.add_camera("main_camera", pos, rot, 1250, 1080, main_fov)

            # translation = left_camera_transform.transform.translation
            # pos = np.array([[-1], [1.219], [1]])
            # rot = np.array([[ 0.7071068,  0.0000000,  0.7071068],
            #                 [ 0.7071068,  0.0000000, -0.7071068],
            #                 [ 0.0000000,  1.0000000,  0.0000000]])
            # panorama.add_camera("left_camera", pos, rot, 1920, 1080, l_r_fov)

            # pos = np.array([[1], [1.219], [1]])
            # rot = np.array([[ 0.7071068,  0.0000000, -0.7071068],
            #                 [-0.7071068,  0.0000000, -0.7071068],
            #                 [ 0.0000000,  1.0000000,  0.0000000]])
            # panorama.add_camera("right_camera", pos, rot, 1920, 1080, l_r_fov)
        

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
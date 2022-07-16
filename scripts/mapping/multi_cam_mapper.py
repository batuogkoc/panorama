#usr/bin/env python3
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../multi_cam_mapping')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../python-utils')))

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
from panorama.srv import RequestMap, RequestMapResponse
from panorama.msg import ImageMap

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

project = None

class MultiCamMapper():
    def __init__(self):
        rospy.init_node("multi_cam_mapper", anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = cv_bridge.CvBridge()
        try:
            node_name = rospy.get_name()
            param_names = rospy.get_param_names()
            param_prefix = node_name + "/"
            self.map_frame_id = rospy.get_param(param_prefix + "map_frame_id", "map")
            self.scale = rospy.get_param(param_prefix + "scale", 0.01)
            self.chunk_size = rospy.get_param(param_prefix + "chunk_size", 200)
            self.extrapolate_htm = rospy.get_param(param_prefix + "extrapolate_htm", False)
            self.continous_output = rospy.get_param(param_prefix + "continous_output", False)
            self.save_location = rospy.get_param(param_prefix + "save_location", "~/Desktop")
            if self.continous_output:
                self.continous_output_frequency = rospy.get_param(param_prefix + "continous_output_frequency", 10)
                self.continous_output_topic = rospy.get_param(param_prefix + "continous_output_topic", "/map_out")
        except rospy.ROSException:
            rospy.logerr("Parameter server error, exiting.")
            raise rospy.ROSInterruptException

        self.project = MultiCamProject(self.scale, self.chunk_size)
        
        camera_names = set()
        for param_name in param_names:
            if param_name.startswith(node_name + "/camera_"):
                param_name = param_name.replace(node_name + "/", "")
                camera_name = param_name.split("/")[0]
                camera_names.add(camera_name)
        self.subscribers = []
        for camera_name in camera_names:
            param_prefix = node_name + "/" + camera_name
            camera_frame_id = fetch_param(param_prefix + "/frame_id")
            camera_image_topic = fetch_param(param_prefix + "/image_topic")
            camera_width = fetch_param(param_prefix + "/width")
            camera_height = fetch_param(param_prefix + "/height")
            camera_depth = fetch_param(param_prefix + "/depth")
            camera_fov = fetch_param(param_prefix + "/fov")
            camera_fov_vertical = rospy.get_param(param_prefix + "/fov_vertical", True)
            camera_image_queue_size = rospy.get_param(param_prefix + "/image_queue_size", 10)
            camera_htm_queue_size = rospy.get_param(param_prefix + "/htm_queue_size", 10)
            camera_orthogonal_roi_width_low = fetch_param(param_prefix + "/orthogonal_roi/width_low")*camera_width
            camera_orthogonal_roi_width_high = fetch_param(param_prefix + "/orthogonal_roi/width_high")*camera_width
            camera_orthogonal_roi_height_low = fetch_param(param_prefix + "/orthogonal_roi/height_low")*camera_height
            camera_orthogonal_roi_height_high = fetch_param(param_prefix + "/orthogonal_roi/height_high")*camera_height
            

            camera = Camera(camera_width, camera_height, m.radians(camera_fov), camera_image_queue_size, camera_htm_queue_size, camera_frame_id, fov_vertical=camera_fov_vertical, depth=camera_depth)
            camera.set_orthogonal_roi(camera_orthogonal_roi_width_low, camera_orthogonal_roi_width_high, camera_orthogonal_roi_height_low, camera_orthogonal_roi_height_high)
            if rospy.has_param(param_prefix+"/mask_area"):
                camera_mask = np.array(rospy.get_param(param_prefix+"/mask_area"))
                camera_mask[:,:,0] = camera_mask[:,:,0] * camera_height
                camera_mask[:,:,1] = camera_mask[:,:,1] * camera_width
                camera.set_mask(camera_mask)
            self.subscribers.append(rospy.Subscriber(camera_image_topic, CompressedImage, self.generate_callback(camera)))
            self.project.add_camera(camera)
        
        rospy.Service("request_map", RequestMap, self.__request_map_callback)
        if self.continous_output:
            self.continous_output_publisher = rospy.Publisher(self.continous_output_topic, ImageMap, queue_size=10)
            def callback(event): 
                self.continous_output_publisher.publish(self.generate_ImageMap())
            rospy.Timer(rospy.Duration(1/self.continous_output_frequency), callback)

    def generate_callback(self, camera: Camera):
        def callback(data):
            frame_id = camera.frame_id
            t = data.header.stamp
            img = self.bridge.compressed_imgmsg_to_cv2(data)
            self.project.cameras[frame_id].add_image_stamped(t, img)
            try:
                transform = self.tf_buffer.lookup_transform(self.map_frame_id, frame_id, rospy.Time(0), rospy.Duration(1.0))
                self.project.cameras[frame_id].add_geometry_msgs_TransformStamped(transform)
                self.project.project_camera(frame_id, extrapolate_htm=self.extrapolate_htm)
            except tf2_ros.TransformException as e:
                rospy.logerr("couldn't get "+frame_id+" tf: "+ str(e))
        return callback

    def shutdown(self):
        rospy.loginfo("Starting image compilation")
        out_img = self.project.get_image(debug=False)
        rospy.loginfo(np.shape(out_img))
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
                cv2.imwrite(os.path.abspath(os.path.join(self.save_location, "out-resized.jpg")), resized_img)
                rospy.loginfo("Saved resized image at " + os.path.abspath(os.path.join(self.save_location, "out-resized.jpg")))
                break
            elif key ==ord("s"):
                cv2.imwrite(os.path.abspath(os.path.join(self.save_location, "out.jpg")), out_img)
                rospy.loginfo("Saved real-size image at " + os.path.abspath(os.path.join(self.save_location, "out.jpg")))
                break
        rospy.loginfo("exiting")

    def __request_map_callback(self, req):
        response = RequestMapResponse()
        response.map = self.generate_ImageMap()

        return response

    def generate_ImageMap(self):
        ret = ImageMap()
        image = self.project.get_image()
        if not image is None:
            ret.map_image = self.bridge.cv2_to_compressed_imgmsg(image)
        else:
            ret.map_image = CompressedImage()
        
        offset = self.project.local_pixels_to_global_meters(np.zeros((3,1)))
        linear_component = self.project.local_pixels_to_global_meters(np.identity(3))-offset
        htm = np.zeros((4,4), dtype=np.float32)
        htm[0:3,0:3] = linear_component
        htm[0:3,3] = offset[0:3,0]
        htm[3,3] = 1

        ret.local_pixels_to_global_meters_htm = htm.reshape(-1)

        return ret

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

def shutdown():
    rospy.loginfo("Starting image compilation")
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
            cv2.imwrite(os.path.abspath(os.path.join(self.save_location, "out-resized.jpg")), resized_img)
            break
        elif key ==ord("s"):
            print(os.path.abspath(os.path.join(self.save_location, "out.jpg")))
            cv2.imwrite(os.path.abspath(os.path.join(self.save_location, "out.jpg")), out_img)
            break
    rospy.loginfo("exiting")


def fetch_param(param_name):
    try:
        return rospy.get_param(param_name)
    except KeyError:
        rospy.logerr(param_name + " not defined, exiting")
        raise Exception


def node():
    mapper = MultiCamMapper()
    rospy.on_shutdown(mapper.shutdown)
    rospy.spin()
    # global project
    # global main_camera_transform
    # global left_camera_transform
    # global right_camera_transform
    # rospy.init_node("multi_cam_mapper", anonymous=True)
    # tf_listener = tf2_ros.TransformListener(tf_buffer)
    # try:
    #     node_name = rospy.get_name()
    #     param_names = rospy.get_param_names()
    #     map_frame_id = rospy.get_param("map_frame_id", "map")
    #     scale = rospy.get_param("scale", 0.01)
    #     chunk_size = rospy.get_param("chunk_size", 200)
    #     continous_output = rospy.get_param("contious_output", False)
    #     if continous_output:
    #         continous_output_frequency = rospy.get_param("continous_output_frequency", 10)
    #         continous_output_topic = rospy.get_param("continous_output_topic", "/map_out")
                
    # except rospy.ROSException:
    #     rospy.logerr("Parameter server error, exiting.")
    #     raise rospy.ROSInterruptException

    # camera_names = set()
    # for param_name in param_names:
    #     if param_name.startswith(node_name + "/camera_"):
    #         param_name = param_name.replace(node_name + "/", "")
    #         camera_name = param_name.split("/")[0]
    #         camera_names.add(camera_name)
    # print(camera_names)
    # cameras = {}
    # for camera_name in camera_names:
    #     param_prefix = node_name + "/" + camera_name
    #     camera_frame_id = fetch_param(param_prefix + "/frame_id")
    #     camera_width = fetch_param(param_prefix + "/width")
    #     camera_height = fetch_param(param_prefix + "/height")
    #     camera_depth = fetch_param(param_prefix + "/depth")
    #     camera_fov = fetch_param(param_prefix + "/fov")
    #     camera_fov_vertical = rospy.get_param(param_prefix + "/fov_vertical", True)
    #     camera_image_queue_size = rospy.get_param(param_prefix + "/image_queue_size", 10)
    #     camera_htm_queue_size = rospy.get_param(param_prefix + "/htm_queue_size", 10)
    #     camera_orthogonal_roi_width_low = fetch_param(param_prefix + "/orthogonal_roi/width_low")*camera_width
    #     camera_orthogonal_roi_width_high = fetch_param(param_prefix + "/orthogonal_roi/width_high")*camera_width
    #     camera_orthogonal_roi_height_low = fetch_param(param_prefix + "/orthogonal_roi/height_low")*camera_height
    #     camera_orthogonal_roi_height_high = fetch_param(param_prefix + "/orthogonal_roi/height_high")*camera_height
    #     if rospy.has_param(param_prefix+"/mask_area"):
    #         pass #fill this up

    #     camera = Camera(camera_width, camera_height, m.radians(camera_fov), camera_image_queue_size, camera_htm_queue_size, camera_frame_id, fov_vertical=camera_fov_vertical, depth=camera_depth)
    #     camera.set_orthogonal_roi(camera_orthogonal_roi_width_low, camera_orthogonal_roi_width_high, camera_orthogonal_roi_height_low, camera_orthogonal_roi_height_high)
    #     cameras[camera_name] = camera
    
    # print(cameras)
    # project = MultiCamProject(0.03)
    # a = 0.58
    # # left_camera = Camera(1920, 1080, m.pi*60/180, 30, 30, "left_camera")
    # # left_camera.set_orthogonal_roi(0, left_camera.width, int(left_camera.height*(a)), left_camera.height)
    # # right_camera = Camera(1920, 1080, m.pi*60/180, 30, 30, "right_camera")
    # # right_camera.set_orthogonal_roi(0, right_camera.width, int(right_camera.height*(a)), right_camera.height)
    # # main_camera = Camera(1250, 1080, m.pi*60/180, 30, 30, "main_camera")
    # # main_camera.set_orthogonal_roi(0, main_camera.width, int(main_camera.height*(a)), main_camera.height)

    # for camera in cameras.values():
    #     project.add_camera(camera)

    # # project.add_camera(left_camera)
    # # project.add_camera(right_camera)
    # # project.add_camera(main_camera)

    # rospy.Subscriber("main_camera/image/compressed", CompressedImage, main_camera_callback)
    # rospy.Subscriber("left_camera/image/compressed", CompressedImage, left_camera_callback)
    # rospy.Subscriber("right_camera/image/compressed", CompressedImage, right_camera_callback)
    # rospy.on_shutdown(shutdown)
    # rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except rospy.ROSInterruptException as e:
        pass
    # except Exception as e:
    #     rospy.logerr("exiting")
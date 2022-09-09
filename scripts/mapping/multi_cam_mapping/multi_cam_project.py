import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
# sys.path.append(os.path.abspath(os.path.join(script_dir, '../../python-utils')))

from multi_cam_mapping.image_append import ImageAppend
from multi_cam_mapping.Map import Map 
import cv2
import numpy as np
import math as m
import rospy
from collections import deque

from geometry_msgs.msg import TransformStamped

from kudrone_py_utils import *
from copy import deepcopy

class Camera():
    def __init__(self, width, height, fov, image_deque_size, htm_deque_size ,frame_id, fov_vertical=True, depth=3):
        self.width = width
        self.width_low = 0
        self.width_high = width

        self.height = height
        self.height_low = 0
        self.height_high = height

        self.depth = depth
        if fov_vertical:
            self.horizontal_fov = vertical_to_horizontal_fov(width, height, fov)
        else:
            self.horizontal_fov = fov
        
        self.__image_deque = deque(maxlen=image_deque_size)

        self.frame_id = frame_id

        self.__htm_deque = deque(maxlen=htm_deque_size)
    


    def set_orthogonal_roi(self, width_low, width_high, height_low, height_high):
        assert 0<=width_low<width_high<=self.width, "incorrect width constraints"
        assert 0<=height_low<height_high<=self.height, "incorrect height constraints"

        self.width_high = width_high
        self.width_low = width_low
        self.height_high = height_high
        self.height_low = height_low

    def calculate_roi_corner_pts(self):
        focal_len = self.width/2/m.tan(self.horizontal_fov/2)
        image_y = np.array([self.height_low, self.height_high]).T
        image_x = np.array([self.width_low, self.width_high]).T

        camera_frame_z = -image_y + self.height/2
        camera_frame_y = -image_x + self.width/2
        return np.vstack([focal_len*np.ones((4)), cartesian_cross_product(camera_frame_y.T, camera_frame_z.T).T])

    def calculate_roi_corner_pts_pixel(self):
        return cartesian_cross_product([self.width_low, self.width_high], [self.height_low,self.height_high]).astype(np.float32).T

    def add_image_stamped(self, stamp, image):
        if hasattr(self, "mask"):
            h, w, d = np.shape(image)
            cv2.fillPoly(image, self.mask, [0 for i in range(d)])
        self.__image_deque.append((stamp, image))
    
    def get_image_stamped(self, offset=-1):
        return self.__image_deque[offset]

    # def get_image_with_closest_stamp(self, time):
    #     closest = self.image_array[self.image_array_index]
    #     for i in range(self.image_array_size):
    #         current=self.image_array[self.image_array_index-i]
    #         if current[0]-time < closest[0]-time:
    #             closest = current
    #     return np.copy(closest)
    
    def get_image_with_closest_stamp_to_htm(self, return_stamped=False):
        closest = self.__image_deque[-1]
        last_htm_stamp = self.__htm_deque[-1][0]
        for current in self.__image_deque:
            if abs(duration_to_sec(current[0]-last_htm_stamp)) < abs(duration_to_sec(closest[0]-last_htm_stamp)):
                closest = current
        if return_stamped:
            return closest
        else:
            return closest[1]

    def add_htm(self, htm, htm_stamp):
        if not (htm_stamp in [a[0] for a in self.__htm_deque]):
            self.__htm_deque.append((htm_stamp, htm))

    def add_geometry_msgs_TransformStamped(self, transform: TransformStamped):
        htm = geometry_msgs_TransformStamped_to_htm(transform)
        stamp = transform.header.stamp
        self.add_htm(htm, stamp)
    
    # def set_htm(self, geometry_msgs_TransformStamped):
    #     self.htm = geometry_msgs_TransformStamped_to_htm(geometry_msgs_TransformStamped)
    #     self.htm_stamp = geometry_msgs_TransformStamped.header.stamp

    def get_htm(self, offset=-1, return_stamped=False):
        if return_stamped:
            return self.__htm_deque[offset]
        else:
            return self.__htm_deque[offset][1]

    def get_extrapolated_htm(self, stamp, return_stamped = False):
        indexes = np.argsort([abs(duration_to_sec(stamp-htm_stamped[0])) for htm_stamped in self.__htm_deque])
        try:
            htm1_stamped = self.__htm_deque[indexes[0]]
            htm2_stamped = self.__htm_deque[indexes[1]]
            htm = (stamp, extrapolate_htm(htm1_stamped[0], htm1_stamped[1], htm2_stamped[0], htm2_stamped[1], stamp))
        except IndexError:
            htm = self.__htm_deque[indexes[0]]
        if return_stamped:
            return htm
        else:
            return htm[1]
    
    def set_mask(self, mask):
        assert len(np.shape(mask)) == 3, "mask must have 3 dimentions, (an array of polygons)"
        self.mask = mask.round().astype(np.int32)


class Panorama_a():
    """
    Class for multi camera panaromic pictures
    Projects images to z=0 plane
    """
    def __init__(self):
        """
        Start a new image append object and initialise it with zero image size. step == how many meters each pixel in the output image represents
        """
        self.image_append = ImageAppend.ImageAppend(0, 0, step=0.01)
        self.cameras = {}
    @staticmethod
    def _focal_length(width, horizontal_fov):
        return width/2/m.tan(horizontal_fov/2)

    @staticmethod
    def _calculateCamImgInitialPos(width, height, horizontal_fov):
        focal_len = width/2/m.tan(horizontal_fov/2)
        # y = np.array([height/2, -(height/2)]).T
        y = np.array([-height/6, -(height/2)]).T
        x = np.array([-width/2, width/2]).T
        return np.vstack([Panorama._cartesian_cross_product(x, y).T, -focal_len*np.ones((4))])

    # @staticmethod
    # def _calculateCamImgInitialPos_with_ROI(width, width_low, width_high, height, height_low, height_high, focal_len):
    #     # y = np.array([height/2, -(height/2)]).T
    #     y = np.array([(height/2-height_low), (height/2-height_high)]).T
    #     x = np.array([width_low-width/2, width_high-width/2]).T
    #     return np.vstack([Panorama._cartesian_cross_product(x, y).T, -focal_len*np.ones((4))])

    @staticmethod
    def _calculateCamImgInitialPos_with_ROI(width, width_low, width_high, height, height_low, height_high, focal_len):
        image_y = np.array([height_low, height_high]).T
        image_x = np.array([width_low, width_high]).T

        camera_frame_z = -image_y + height/2
        camera_frame_y = -image_x + width/2
        return np.vstack([focal_len*np.ones((4)), Panorama._cartesian_cross_product(camera_frame_y.T, camera_frame_z.T).T])


    @staticmethod
    def _project_points(corner_pts, camera_position):
        xc = camera_position[0][0]
        yc = camera_position[1][0]
        zc = camera_position[2][0]

        denom = zc - corner_pts[2]

        transform1 = np.array([[zc, 0, -xc],
                            [0, zc, -yc]])

        corner_pts = np.matmul(transform1, corner_pts)
        return np.divide(corner_pts, [denom, denom])

    @staticmethod
    def _cartesian_cross_product(x,y):
        cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
        return cross_product
    def add_camera(self, camera_name, camera_pos, camera_rot, img_width, img_height, horizontal_fov):
        """Adds a camera to the list of cameras

        Args:
            camera_name (string): name of the camera to add. ex: "scripts"front_left"
            camera_pos (3x1 np array): how many meters the camera frame is offset from the base frame. base frame: +z value is height from ground. +x is towards left of output frame. +y is towards top of output frame.
            camera_rot (3x3 np array): rotation matrix from base frame to camera frame. camera frame: +x is towards left of frame. +y is towards top of frame. -z is towards the center of the frame
            img_width (int): width of image
            img_height (int): height of image
            horizontal_fov (float): horizontal fov in radians
        """

        self.cameras[camera_name] = {"pos":camera_pos,
                                     "rot": camera_rot,
                                     "width": img_width,
                                     "height": img_height,
                                     "horizontal_fov": horizontal_fov,
                                     "corner_pts_start": Panorama._calculateCamImgInitialPos(img_width, img_height, horizontal_fov)}

    def remove_camera(self, camera_name):
        """Removes a camera from the list of cameras

        Args:
            camera_name (string): name of the camera to remove
        """
        del(self.cameras[camera_name])

    def update_camera_pos_rot(self, camera_name, camera_pos, camera_rot):
        self.cameras[camera_name]["pos"] = camera_pos
        self.cameras[camera_name]["rot"] = camera_rot
    
    def update_camera_img(self, camera_name, camera_img):
        camera_pos = self.cameras[camera_name]["pos"]
        camera_rot = self.cameras[camera_name]["rot"]
        width = self.cameras[camera_name]["width"]
        height = self.cameras[camera_name]["height"]
        horizontal_fov = self.cameras[camera_name]["horizontal_fov"]

        (img_height, img_width, _) = np.shape(camera_img)
        from_pts = self._cartesian_cross_product([0,img_width-1], [int(img_height*(2/3)), img_height-1]).astype(np.float32).T

        x_min = np.min(from_pts[:][0])
        x_max = np.max(from_pts[:][0])
        y_min = np.min(from_pts[:][1])
        y_max = np.max(from_pts[:][1])
        
        corner_pts_start = self._calculateCamImgInitialPos_with_ROI(width, x_min, x_max, height, y_min, y_max, self._focal_length(width, horizontal_fov))
        # corner_pts_start = self._calculateCamImgInitialPos(width, height, horizontal_fov)

        corner_pts = np.matmul(camera_rot, corner_pts_start) + camera_pos

        projected_points = Panorama._project_points(corner_pts, camera_pos)
        
        to_pts_abs = self.image_append.local_meter_to_local_pixel_coords(projected_points)
        self.image_append.append(camera_img, from_pts, to_pts_abs)
        # self.image_append.project(camera_img, projected_points)
    
    def get_output_img(self):
        return self.image_append.image

    def clear_img(self):
        self.image_append.clear_img()

if __name__ == "__main__":
    panorama = Panorama()

    #front camera 1 meter ahead, 1 meter above looking forward with 45 degree angle, 1.2 radian horizontal fov, 600x400 image
    pos =  np.array([[0,1,1]]).T
    rot = np.array([[1.0000000,  0.0000000,  0.0000000],
                    [0.0000000,  0.7071068, -0.7071068],
                    [0.0000000,  0.7071068,  0.7071068]])
    panorama.add_camera("front_camera", pos, rot, 600, 400, 1.2)

    #left camera 1 meter left, 0.5 meter above looking left with 45 degree angle 1.2 radian horizontal fov, 600x400 image
    pos =  np.array([[-1,0,0.5]]).T
    rot = np.array([[ 0.7071068,  0.0000000,  0.7071068],
                    [ 0.0000000,  1.0000000,  0.0000000],
                    [-0.7071068,  0.0000000,  0.7071068]])
    panorama.add_camera("left_camera", pos, rot, 600, 400, 1.2)

    #right camera 1 meter right, 0.5 meter above looking left with 45 degree angle 1.2 radian horizontal fov, 600x400 image
    pos =  np.array([[1,0,0.5]]).T
    rot = np.array([[ 0.7071068,  0.0000000, -0.7071068],
                    [ 0.0000000,  1.0000000,  0.0000000],
                    [ 0.7071068,  0.0000000,  0.7071068]])
    panorama.add_camera("right_camera", pos, rot, 600, 400, 1.2)

    blank_img = np.zeros((400, 600, 3)).astype(np.float32) #black 600x400 img

    #front camera sees all red
    front_img = blank_img.copy()
    front_img[:] = (0,0,255)
    
    #right camera sees all green
    right_img = blank_img.copy()
    right_img[:] = (0,255,0)

    #left camera sees all blue
    left_img = blank_img.copy()
    left_img[:] = (255,0,0)

    panorama.clear_img()#optional. If not done so, new images will override the pixel values in the regions that will be occupied. If the base frame is moved as the car moves, a mapping effect can be achieved if this line is ommitted.
    panorama.update_camera_img("front_camera", front_img)
    panorama.update_camera_img("right_camera", right_img)
    panorama.update_camera_img("left_camera", left_img)

    cv2.imwrite("img.jpg", panorama.get_output_img())


class MultiCamProject(Map):
    def __init__(self, step=0.01, chunk_size=200):
        self.image_append = ImageAppend(chunk_size=chunk_size, depth=3)
        self.step = step
        self.cameras = {}
    
    def add_camera(self, camera):
        frame_id = camera.frame_id
        self.cameras[frame_id] = camera

    def remove_camera(self, camera_frame_id):
        del(self.cameras[camera_name])

    def project_camera(self, camera_frame_id, extrapolate_htm=False):
        camera = self.cameras[camera_frame_id]
        roi_corner_pts = camera.calculate_roi_corner_pts()
        from_pts = camera.calculate_roi_corner_pts_pixel()

        image_stamp, image = camera.get_image_with_closest_stamp_to_htm(return_stamped=True)

        if extrapolate_htm:
            camera_htm = camera.get_extrapolated_htm(image_stamp)
        else:
            camera_htm = camera.get_htm()

        roi_corner_pts_transformed = np.matmul(camera_htm, np.vstack((roi_corner_pts, np.ones(np.shape(roi_corner_pts)[1]))))[0:3]

        camera_pos, camera_rot = htm_to_pos_rot(camera.get_htm())
        roi_corner_pts_projected = project_points(roi_corner_pts_transformed, camera_pos)
        
        to_pts = self.global_meters_to_global_pixels(roi_corner_pts_projected)
        self.image_append.append(image, from_pts, to_pts)
    
    def project_all_cameras(self, extrapolate_htm=False):
        for camera_frame_id in self.cameras.keys():
            self.project_camera(camera_frame_id, extrapolate_htm=extrapolate_htm)
    
    def get_image(self, debug=False):
        return self.image_append.get_image(debug=debug)

    def clear_img(self):
        self.image_append.clear_img()
    
    def global_pixels_to_global_meters(self, global_pixel_coordinates):
        assert np.shape(global_pixel_coordinates)[0] == 2 or np.shape(global_pixel_coordinates)[0] == 3, "invalid global_pixel_coordinates shape, expected 2xn or 3xn"
        if np.shape(global_pixel_coordinates)[0] == 3:
            transform = np.array([[ 1.0000000,  0.0000000,  0.0000000],
                                  [ 0.0000000, -1.0000000,  0.0000000],
                                  [ 0.0000000,  0.0000000, -1.0000000]])
        elif np.shape(global_pixel_coordinates)[0] == 2:
            transform = np.array([[ 1.0000000,  0.0000000],
                                  [ 0.0000000, -1.0000000]])
        global_pixel_coordinates_new_axes = np.matmul(np.linalg.inv(transform), global_pixel_coordinates)

        global_meter_coordinates = global_pixel_coordinates_new_axes*self.step

        return global_meter_coordinates.astype(np.float32)

    def global_meters_to_global_pixels(self, global_meter_coordinates):
        assert np.shape(global_meter_coordinates)[0] == 2 or np.shape(global_meter_coordinates)[0] == 3, "invalid global_meter_coordinates shape, expected 2xn or 3xn"
        if np.shape(global_meter_coordinates)[0] == 3:
            transform = np.array([[ 1.0000000,  0.0000000,  0.0000000],
                                  [ 0.0000000, -1.0000000,  0.0000000],
                                  [ 0.0000000,  0.0000000, -1.0000000]])
        elif np.shape(global_meter_coordinates)[0] == 2:
            transform = np.array([[ 1.0000000,  0.0000000],
                                  [ 0.0000000, -1.0000000]])
        global_meter_coordinates_new_axes = np.matmul(transform, global_meter_coordinates)

        global_pixel_coordinates = global_meter_coordinates_new_axes/self.step
        return global_pixel_coordinates.astype(np.float32)

    def local_pixels_to_global_meters(self, local_pixel_coordinates):
        assert np.shape(local_pixel_coordinates)[0] == 2 or np.shape(local_pixel_coordinates)[0] == 3, "invalid local_pixel_coordinates shape, expected 2xn or 3xn"
        try:
            global_pixel_coordinates = np.copy(local_pixel_coordinates)
            global_pixel_coordinates[0:2,:] += np.array([[self.image_append.x_min],[self.image_append.y_min]])
            return self.global_pixels_to_global_meters(global_pixel_coordinates)
        except AttributeError:
            raise AttributeError("No image has been added to image_append, local pixel coordinates haven't been initialised")

    def global_meters_to_local_pixels(self, global_meter_coordinates):
        assert np.shape(global_meter_coordinates)[0] == 2 or np.shape(global_meter_coordinates)[0] == 3, "invalid global_meter_coordinates shape, expected 2xn or 3xn"
        global_pixel_coordinates = self.global_meters_to_global_pixels(global_meter_coordinates)
        try:
            local_pixel_coordinates = np.copy(global_meter_coordinates)
            local_pixel_coordinates[0:2,:] -= np.array([[self.image_append.x_min],[self.image_append.y_min]])
            return local_pixel_coordinates
        except AttributeError:
            raise AttributeError("No image has been added to image_append, local pixel coordinates haven't been initialised")
        


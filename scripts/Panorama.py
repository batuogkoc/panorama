import ImageAppend
import cv2
import numpy as np
import math as m

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

class Panorama():
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
        # y = np.array([height/2, -(height/2)]).T
        z = np.array([(height/2-height_low), (height/2-height_high)]).T
        y = np.array([(width_low-width/2), (width_high-width/2)]).T
        # return np.vstack([Panorama._cartesian_cross_product(x, y).T, -focal_len*np.ones((4))])
        return np.vstack([-focal_len*np.ones((4)), Panorama._cartesian_cross_product(y, z).T])


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



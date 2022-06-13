#usr/bin/env python
import Panorama
import sys
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/python-utils')
from utils import *
import numpy as np

def project_pixels_tf(pixel_coords, camera_geometry_msgs_TransformStamped, image_height, image_width, horizontal_fov):
    assert np.shape(pixel_coords)[0] == 2, f"pixel_coords must be of size (2,n), not {np.shape(pixel_coords)}"
    pixel_count = np.shape(pixel_coords)[1]
    camera_htm = geometry_msgs_TransformStamped_to_htm(camera_geometry_msgs_TransformStamped)

    focal_len = image_width/2/m.tan(horizontal_fov/2)
    image_y = pixel_coords[1]
    image_x = pixel_coords[0]

    camera_frame_z = -image_y + image_height/2
    camera_frame_y = -image_x + image_width/2
    pixel_coords_3d = np.vstack([focal_len*np.ones((pixel_count)), camera_frame_y, camera_frame_z])
    pixel_coords_3d_transformed = np.matmul(camera_htm, np.vstack((pixel_coords_3d, np.ones(pixel_count))))[0:3]

    camera_pos, camera_rot = htm_to_pos_rot(camera_htm)
    return project_points(pixel_coords_3d_transformed, camera_pos)

def project_pixels_htm(pixel_coords, camera_htm, image_height, image_width, horizontal_fov):
    assert np.shape(pixel_coords)[0] == 2, f"pixel_coords must be of size (2,n), not {np.shape(pixel_coords)}"
    pixel_count = np.shape(pixel_coords)[1]

    focal_len = image_width/2/m.tan(horizontal_fov/2)
    image_y = pixel_coords[1]
    image_x = pixel_coords[0]

    camera_frame_z = -image_y + image_height/2
    camera_frame_y = -image_x + image_width/2
    pixel_coords_3d = np.vstack([focal_len*np.ones((pixel_count)), camera_frame_y, camera_frame_z])
    pixel_coords_3d_transformed = np.matmul(camera_htm, np.vstack((pixel_coords_3d, np.ones(pixel_count))))[0:3]

    camera_pos, camera_rot = htm_to_pos_rot(camera_htm)
    return project_points(pixel_coords_3d_transformed, camera_pos)

if __name__ == "__main__":
    #Camera 1 meter in the air, looking straight down. top of image is towards +x, left of image is towards -y. Camera pointing at -z (In the map frame)
    camera_htm = np.array([[0,  0, 1, 0],
                           [0,  1, 0, 0],
                           [-1, 0, 0, 1],
                           [0,  0, 0, 0]])

    #first point is the top left corner of screen
    #second point is the top right corner of screen
    #third point is the middle of the screen
    #fourth point is bottom right corner of screen
    points = np.array([[0,100, 50,100],
                       [0,  0, 50,100]])
    #100x100 image, with a horizontal field of view of 90 degrees
    print(project_pixels_htm(points, camera_htm, 100, 100, m.pi/2))

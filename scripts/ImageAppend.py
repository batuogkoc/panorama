#!/usr/bin/env python3
import cv2
import numpy as np
import math as m

class ImageAppend:
    def __init__(this, width, height, step = 0.2, depth = 3):
        this.step = step
        this.width = width
        this.height = height
        this.depth = depth
        this.map_corner_coords = np.array([[0, 0]], dtype=np.int)  #coordinates of where the map top left corner is in local pixel coordinate frame
        this.image = np.zeros((height, width, depth))

    @staticmethod
    def _cartesian_cross_product(x,y):
        cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
        return cross_product

    def updateImage(this, new_img):
        (img_height, img_width, img_depth) = np.shape(new_img)
        this.width = img_width
        this.height = img_height
        this.depth = img_depth
        this.image = new_img

    # def local_meter_to_local_pixel_coords(this, local_meter_coords):
    #     local_meter_coords_temp = np.copy(local_meter_coords)

    #     local_meter_coords_temp[1] = -local_meter_coords_temp[1]

    #     local_pixel_coords = local_meter_coords_temp/this.step

    #     return local_pixel_coords.astype(np.float32)

    def local_meter_to_local_pixel_coords(this, local_meter_coords):
        assert np.shape(local_meter_coords)[0] == 2 or np.shape(local_meter_coords)[0] == 3, "invalid local_meter_coords shape, expected 2xn or 3xn"
        local_meter_coords_temp = np.copy(local_meter_coords)
        if np.shape(local_meter_coords)[0] == 3:
            transform = np.array([[ 0.0000000, -1.0000000,  0.0000000],
                                  [-1.0000000,  0.0000000,  0.0000000],
                                  [ 0.0000000,  0.0000000,  1.0000000]])
        elif np.shape(local_meter_coords)[0] == 2:
            transform = np.array([[ 0.0000000, -1.0000000],
                                  [-1.0000000,  0.0000000]])
        local_meter_coords_temp = np.matmul(transform, local_meter_coords)

        local_pixel_coords = local_meter_coords_temp/this.step

        return local_pixel_coords.astype(np.float32)

    def clear_img(self):
        self.image = np.zeros((self.height, self.width, self.depth))

    
    def project(this, img, projected_points):
        (img_height, img_width, _) = np.shape(img)
        to_pts_abs = this.local_meter_to_local_pixel_coords(projected_points)
        from_pts = this._cartesian_cross_product([0,img_width-1], [int(img_height*(2/3)), img_height-1]).astype(np.float32).T
        this.append(img, from_pts, to_pts_abs)

    def append(this, img, from_pts, to_pts, mask=True, extend_mask=True):
        from_pts = from_pts.T
        corner_pixel_values = to_pts.T

        #round the coordinates of the corners which are in home center pixel coordinate frame
        corner_pixel_values = np.round(corner_pixel_values).astype(np.int)
        (img_height, img_width, _) = np.shape(img)
        
        #get boundaries of the image to add
        x_min_img = np.min(corner_pixel_values.T[0])
        x_max_img = np.max(corner_pixel_values.T[0])
        y_min_img = np.min(corner_pixel_values.T[1])
        y_max_img = np.max(corner_pixel_values.T[1])
        
        #set the image width to span from the lowest x to the highest. Same with the height
        new_width = max(this.width+this.map_corner_coords[0][0], x_max_img+1) - min(this.map_corner_coords[0][0], x_min_img)
        new_height = max(this.height+this.map_corner_coords[0][1], y_max_img+1) - min(this.map_corner_coords[0][1], y_min_img)

        #initialise empty image to copy the current image into. These are in the form of the updated image pixel coordinates. 
        old_img_new_index = np.zeros((new_height, new_width, this.depth))
        
        #save the old map corner coordinates
        old_map_corner_coords = np.copy(this.map_corner_coords)
        #new map corner coordinates are the lower of the x and y values of old map_corner_coords and top left corner of image to stitch
        this.map_corner_coords = np.array([[min(this.map_corner_coords[0][0], x_min_img), min(this.map_corner_coords[0][1], y_min_img)]], dtype=np.int)

        #how much to offset new image
        offset = this.map_corner_coords
        #how much to offset old image
        offset_old = this.map_corner_coords-old_map_corner_coords

        #copy every pixel from old image into the empty old_img_new_index picture which has its map_corner_coords at the new map_corner_coords
        old_img_new_index[-offset_old[0][1]:this.height-offset_old[0][1],-offset_old[0][0]:this.width-offset_old[0][0]] = this.image[:,:]
        old_img_new_index = old_img_new_index.astype(np.float32)#needed for opencv for some reason

        #corners of the input image
        # from_pts = this._cartesian_cross_product([0,img_width-1], [0, img_height-1]).astype(np.float32)
        # from_pts = this._cartesian_cross_product([0,img_width-1], [int(img_height*(2/3)), img_height-1]).astype(np.float32)

        #where the corners go (in pixel coordinates)
        to_pts = ((corner_pixel_values.T - this.map_corner_coords.T).T).astype(np.float32)

        if not x_max_img == x_min_img and not y_max_img == y_min_img:
            #project the image only if the pixel values arent the same. the coordinate space is the same as old_img_new_index
            perspective_matrix = cv2.getPerspectiveTransform(from_pts, to_pts)
            new_img_new_index = cv2.warpPerspective(img, perspective_matrix, (new_width,new_height))
            if mask:
                mask = np.zeros((list(np.shape(new_img_new_index)[0:2]) + [1]), dtype=np.uint8)
                poly_pts = np.copy(to_pts).astype(np.int32)
                poly_pts[2:4] = poly_pts[3:1:-1]
                x_0 = poly_pts[0][0]
                y_0 = poly_pts[0][1]
                x_1 = poly_pts[1][0]
                y_1 = poly_pts[1][1]
                if extend_mask:
                    dx_1 = x_1-x_0
                    dy_1 = -(y_1-y_0)
                    if dx_1 > 0:
                        if dy_1 > 0:
                            midpoint = [x_0, y_1]
                        else:
                            midpoint = [x_1, y_0]
                    else:
                        if dy_1 > 0:
                            midpoint = [x_1, y_0]
                        else:
                            midpoint = [x_0, y_1]
                    poly_pts = np.vstack((poly_pts[0], midpoint, poly_pts[1:]))
                cv2.fillPoly(mask, np.int32([poly_pts]), (255))
                new_img_new_index = cv2.bitwise_and(new_img_new_index, new_img_new_index, mask=mask)    
        else:
            new_img_new_index = np.zeros((new_height, new_width, this.depth)).astype(np.float32)

        #create a mask to black out the region where the new image will fit in the old image
        new_img_new_index_gray = cv2.cvtColor(new_img_new_index, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(new_img_new_index_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask.astype(np.uint8))

        #black out area where the new image will fit in the old image
        old_img_new_index = cv2.bitwise_and(old_img_new_index, old_img_new_index, mask=mask_inv.astype(np.uint8))

        #stitch images
        ret = (old_img_new_index.astype(np.uint8) + new_img_new_index.astype(np.uint8))

        this.updateImage(ret)
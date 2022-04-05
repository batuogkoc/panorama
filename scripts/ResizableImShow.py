import cv2
import numpy as np

def imshow_r(window_name, img, window_size):
    img_height, img_width, _ = np.shape(img)
    window_width, window_height = window_size
    window_aspect = window_width/window_height
    img_aspect = img_width/img_height
    if img_aspect>window_aspect:
        out_width = window_width
        out_height = int(out_width/img_aspect)
    else:
        out_height = window_height
        out_width = int(out_height*img_aspect)
    resized = cv2.resize(img, (out_width, out_height))
    cv2.imshow(window_name, resized)
    return resized
# std_msgs/Header header
# geographic_msgs/GeoPoint origin_location #where the origin of "global meters" (i.e. coordinate frame dictated by "map_frame_id") lies in the world
float32[16] local_pixels_to_global_meters_htm #transforms a pixel on the image into the coordinate frame dictated by "map_frame_id", row-major order
sensor_msgs/CompressedImage map_image
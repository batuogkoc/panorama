import sys
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/mapping')
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/localisation/map_annotator')
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/python-utils')
from Graph import *
import rospy
import numpy as np
import Panorama
import tf2_ros
import cv2
from utils import *
import pickle
import math as m
from copy import deepcopy

map_path = "/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/map/map.panorama.pickle"
directed = True

def node():
    with open(map_path, "rb") as f:
        panorama = pickle.load(f)
    rospy.init_node("localisation")
    global tf_buffer
    global tf_listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    map_image_original = cv2.dilate(deepcopy(panorama.get_output_img()), (3,3), iterations=2)
    graph_opened = False
    with open("/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/localisation/graph.pickle", "rb") as f:
        loaded_object = pickle.load(f) 
        graph_opened = True
        if type(loaded_object) == Graph:
            graph=loaded_object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    global out
    # out = cv2.VideoWriter("output.avi", fourcc, 15, (1200, 1400))
    is_vehicle_reversed = True
    while True:
        map_image = np.copy(map_image_original)
        try:
            vehicle_transform = tf_buffer.lookup_transform("odom", "map", rospy.Time(0))
        except tf2_ros.LookupException as e:
            rospy.loginfo(f"waiting for tf: {e}")
            rospy.sleep(0.5)
            continue
        except Exception as e:
            rospy.logerr(e)
        
        raw_htm = geometry_msgs_TransformStamped_to_htm(vehicle_transform)
        transform = np.array([[0,1,0,0],
                            [-1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        htm = np.matmul(transform, raw_htm)
        htm_only_z = only_z_rot(htm)
        vehicle_pos, vehicle_rot = htm_to_pos_rot(htm_only_z)
        vehicle_rot = vehicle_rot[0:2,0:2]
        vehicle_pos = vehicle_pos[0:2] + np.array([[-53,1.5]]).T

        marker = np.array([[0,0],
                           [-0.5,-0.5],
                           [1,0],
                           [-0.5, 0.5]]).T*2
        marker = np.matmul(vehicle_rot, marker)
        marker_final = marker + vehicle_pos
        vehicle_pos = panorama.meters_to_pixels(vehicle_pos)
        marker_final = np.round(panorama.meters_to_pixels(marker_final)).astype(np.int32)
        marker_final = marker_final.T.astype(np.int32)
       

        vehicle_pos_pixels = vehicle_pos.round().astype(int)

        # current_travelable_element, next_travelable_elements = localise_in_graph(graph, vehicle_pos.round().astype(int))
        
        current_travelable_element =graph.find_closest_element(vehicle_pos_pixels, strictly_travelable=True)
        if type(current_travelable_element) == Edge:
            edge_vector=current_travelable_element.nodes[1].position-current_travelable_element.nodes[0].position
            dot = np.matmul(np.matmul(vehicle_rot, np.array([[1,0]]).T), edge_vector.T)#inner matmul vreates a unit vector in the direction of car, the second matmul is a dot product between where the car is headed and where the travelable road element is going
            if dot[0][0]<0:
                is_vehicle_reversed = True #if the dot product is negative we are going in the wrong direction
            else:
                is_vehicle_reversed = False

        legal_nodes, illegal_nodes = current_travelable_element.travelable_nodes(invert_direction=is_vehicle_reversed)
        print(f"\nLegal node count: {len(legal_nodes)}")
        print("Possible legal nodes to travel to")
        for node in legal_nodes:
            print(panorama.pixel_to_meters(node.position))
        
        special_elements = [current_travelable_element]+legal_nodes #+ next_travelable_elements
        special_element_colors = [(0,255,255)]+[(255,0,255) for i in range(len(legal_nodes))]# + [(255,0,255) for i in range(len(next_travelable_elements))]
        map_image = graph.draw(map_image, special_elements, special_element_colors, draw_order=[Edge, Intersection, Node])

        map_image = cv2.fillPoly(map_image, [marker_final], (255,255,0), lineType=cv2.LINE_AA)
        img  = imshow_r("win", map_image, (600,900))
        # out.write(img)
        if cv2.waitKey(1) == ord("q"):
            return
        

if __name__=="__main__":
    try:
        node()
    except Exception as e:
        rospy.loginfo("Quitting: " + str(e))
    finally:
        cv2.destroyAllWindows()
        # out.release()
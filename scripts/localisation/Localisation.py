import sys
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/mapping')
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/python_tests/map_annotator')
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/python-utils')
import rospy
import numpy as np
import Panorama
import tf2_ros
import cv2
from utils import *
import pickle
import math as m
from copy import deepcopy
import map_annotator.annotator as an

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
        nodes, edges = pickle.load(f) 
        graph_opened = True
        if not directed:
            for edge in edges:
                new_edge = (edge[1], edge[0])
                if not new_edge in edges:
                    edges.append(new_edge)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("output.avi", fourcc, 15, (1200, 1400))
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
       

        vehicle_pos_pixels = (int(round(vehicle_pos[0][0])), int(round(vehicle_pos[1][0])))
        current_edge, next_edges = an.localise_in_graph(vehicle_pos_pixels, edges)
        rospy.loginfo(f"\n\nAvailable roads: {len(next_edges)}")
        rospy.loginfo("Possible nodes to travel to")
        for edge in next_edges:
            rospy.loginfo(edge[1])
        map_image = an.draw_edges(map_image, edges, special_edges=[current_edge]+next_edges, special_edge_colors=[(0,255,255)]+[(255,0,255) for i in range(len(next_edges))], use_arrows=directed)
        map_image = an.draw_nodes(map_image, nodes)
        map_image = cv2.fillPoly(map_image, [marker_final], (255,255,0), lineType=cv2.LINE_AA)
        img  = imshow_r("win", map_image, (1200,1400))
        out.write(img)
        if cv2.waitKey(1) == ord("q"):
            return
        

if __name__=="__main__":
    try:
        node()
    except Exception as e:
        rospy.loginfo("Quitting: " + e)
    finally:
        cv2.destroyAllWindows()
        out.release()
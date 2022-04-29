import cv2
from enum import Enum
import numpy as np
import math as m
import pickle
from Graph import *


#USER PARAMETERS
image_path = "/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/localisation/map.png" #the path to the input image
win_scale = 0.6 #how much the input image will be scaled

graph_load_dir = "/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/localisation/graph.pickle" #where to load the previous graph from, if you will load. Won't do anytinh if there is no such file
graph_save_dir = "/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/localisation/graph.pickle" #where to save the graph

use_middle_mouse = False #whether to use middle mouse for clicks instead of left mouse button. Allows you to pan when zoomed but is a bit cumbersome. 

node_add_mode = "n" #switch to node addition mode. Clicks will add a node
edge_add_mode = "e" #switch to edge addition mode. Consecutive clicks will add an edge between closest nodes at the time of the clicks
delete_mode = "d" #switch to deletion mode. Clicks will delete the closest node or edge
clear_graph = "c" #clears graph
localise_mode = "l" #localises the mouse in the graph
intersection_mode = "i" #add intersection. start from a node that has right to enter the intersection, go clockwise
quit_button = "q" #quits and saves the constructed graph

u_turn_at_intersections = False #if set to true, the intersection mode will generate a u turn node at intersections
#END USER PARAMETERS

win_name = "Annotator"

if not use_middle_mouse:
    click_down = cv2.EVENT_LBUTTONDOWN
    click_up = cv2.EVENT_LBUTTONUP
else:
    click_down = cv2.EVENT_MBUTTONDOWN
    click_up = cv2.EVENT_MBUTTONUP

state = 0

global graph
graph = Graph()

mouse_coords = None

special_elements = []
special_element_colors = []

new_edge_first_node=None
intersection = []

resizable_image = None


class ResizableImage():
    def __init__(self, scale):
        self.scale = scale
    
    def resize(self, img):
        img = np.copy(img)
        img_height, img_width, img_depth = np.shape(img)
        resized = cv2.resize(img, (int(img_width*self.scale), int(img_height*self.scale)))
        return resized

    # def in_to_out_point(self, point):
    #     x, y = point
    #     return (x-self.in_height//2+self.out_height//2,y-self.in_width//2+self.out_width//2)
    
    def out_to_in_point(self, point):
        x, y = point
        return (int(x/self.scale), int(y/self.scale))


def click(event, x, y, flags, param):
    global graph
    global new_edge_first_node
    global special_elements
    global special_element_colors

    global resizable_image
    global mouse_coords
    global intersection
    point = resizable_image.out_to_in_point((x,y))
    point = np.array([[point[0]], [point[1]]], dtype=int)
    mouse_coords = point
    if state == 0:
        special_elements = []
        special_element_colors = []
        if event == click_down:
            pass
        elif event == click_up:
            graph.add_element(Node(mouse_coords))

    elif state == 1:
        closest_element = graph.find_closest_element(mouse_coords, element_type=Node)
        if new_edge_first_node == None:
            special_elements = [closest_element]
            special_element_colors = [(0,255,255)]
            if event == click_down:
                pass
            elif event == click_up:
                new_edge_first_node = closest_element
        else:
            special_elements = [new_edge_first_node, closest_element]
            special_element_colors = [(255,255,0),(0,255,255)]
            if event == click_down:
                pass
            elif event == click_up:
                if new_edge_first_node != closest_element:
                    graph.add_element(Edge(new_edge_first_node, closest_element))
                    new_edge_first_node = None
                else:
                    new_edge_first_node = None


    elif state == 2:
        closest_element = graph.find_closest_element(mouse_coords)
        special_elements = [closest_element]
        special_element_colors = [(0,0,255)]
        if event == click_down:
            pass
        elif event == click_up:
            graph.remove_element(closest_element)
    
    elif state == 6:
        closest_node=graph.find_closest_element(mouse_coords, element_type=Node)
        special_elements=[closest_node]+intersection
        special_element_colors=[(255,0,255)]+[(255,255,0) if i%2==0 else (0,255,255) for i in range(len(intersection))]
        if event == click_up:
            intersection.append(closest_node)

if __name__ == "__main__":

    image = cv2.imread(image_path)
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, click)
    resizable_image = ResizableImage(win_scale)
    if graph_load_dir != "":
        try:
            with open(graph_load_dir, "rb") as file:
                loaded_object = pickle.load(file)
                if type(loaded_object) == Graph:
                    graph = loaded_object
        except FileNotFoundError:
            pass

    while True:
        frame = np.copy(image)
        # if state == 0:
        #     frame = draw_edges(frame, edges)
        #     frame = draw_nodes(frame, nodes)
        # elif state == 1:
        #     frame = draw_edges(frame, edges)
        #     frame = draw_nodes(frame, nodes, special_nodes=[closest_node, new_edge_first_point], special_node_colors=[(0,255,255), (255,255,0)])
        # elif state == 2:
        #     if closest_edge != None:
        #         frame = draw_edges(frame, edges, special_edges=[closest_edge], special_edge_colors=[(0,0,255)])
        #     else:
        #         frame = draw_edges(frame, edges)
                
        #     if closest_node != None:
        #         frame = draw_nodes(frame, nodes, special_nodes=[closest_node], special_node_colors=[(0,0,255)])
        #     else:
        #         frame = draw_nodes(frame, nodes)
        
        if state == 3:
            height, width, depth = np.shape(frame)
            frame = putTextCenter(frame, "Quit?",(width//2, height//2), cv2.FONT_HERSHEY_COMPLEX, 5, (0,0,255), thickness=10, lineType=cv2.LINE_AA)
        elif state == 4:
            height, width, depth = np.shape(frame)
            frame = putTextCenter(frame, "Clear?",(width//2, height//2), cv2.FONT_HERSHEY_COMPLEX, 5, (0,0,255), thickness=10, lineType=cv2.LINE_AA)
        elif state == 5:
            current_travelable_element, next_travelable_elements = localise_in_graph(graph, mouse_coords)
            legal_nodes = current_travelable_element.exiting_nodes()
            print(f"\nLegal node count: {len(legal_nodes)}")
            print("Possible legal nodes to travel to")
            for node in legal_nodes:
                print(node.position)
            
            special_elements = [current_travelable_element]+legal_nodes + next_travelable_elements
            special_element_colors = [(0,255,255)]+[(255,0,255) for i in range(len(special_elements))] + [(255,0,255) for i in range(len(next_travelable_elements))]

        # elif state == 6:
        #     frame = draw_edges(frame, edges)
        #     frame = draw_nodes(frame, nodes, special_nodes=[closest_node]+ intersection, special_node_colors=[(255,0,255)]+[(255,255,0) if i%2==0 else (0,255,255) for i in range(len(intersection))])

        if state != 1:
            new_edge_first_node = None
        if state != 6:
            intersection = []

        frame = graph.draw(frame, special_elements=special_elements, special_element_colors=special_element_colors, draw_order=[Intersection, Edge, Node])
        frame = resizable_image.resize(frame)
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) 
        if key == ord(node_add_mode):
            state = 0 #node adding mode
        elif key == ord(edge_add_mode):
            state = 1 #edge adding mode
        elif key == ord(delete_mode):
            state = 2 #node/edge deletion mode
        elif key == ord(quit_button):
            if state==3:
                break
            state = 3
        elif key == ord(clear_graph):
            if state == 4:
                graph.clear()
                state = 0
            else:
                state = 4
        elif key == ord(localise_mode):
            state = 5
        elif key == ord(intersection_mode):
            if state==6 and intersection != []:
                try:
                    graph.add_element(Intersection(intersection))
                except AssertionError:
                    pass
                intersection = []
            state=6
        elif key == -1:
            continue
        else:
            state = 0

    cv2.destroyAllWindows()
    if graph_save_dir != "":
        with open(graph_save_dir,'wb') as file:
            pickle.dump(graph, file)
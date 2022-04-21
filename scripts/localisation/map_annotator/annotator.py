import cv2
from enum import Enum
import numpy as np
import math as m
import pickle

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

image = cv2.imread(image_path)
if not use_middle_mouse:
    click_down = cv2.EVENT_LBUTTONDOWN
    click_up = cv2.EVENT_LBUTTONUP
else:
    click_down = cv2.EVENT_MBUTTONDOWN
    click_up = cv2.EVENT_MBUTTONUP

state = 0
nodes = []
edges = []

new_edge_first_point=None
closest_node = None
closest_edge = None
mouse_coords = None

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
def linear_transform(inp, in_min, in_max, out_min, out_max):
     return max(out_min, min(out_max, (inp-in_min)/(in_max-in_min) * (out_max-out_min) + out_min))

def distance(point1, point2):
    return m.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def distance_edge_center(point, edge):
    edge_center = ((edge[0][0]+edge[1][0])/2, (edge[0][1]+edge[1][1])/2)
    return distance(edge_center, point)

def find_closest_node(point, nodes):
    if len(nodes) > 0:
        closest = nodes[0]
        old_dist = distance(point, closest)
        for node in nodes:
            new_dist=distance(point, node)
            if new_dist<old_dist:
                old_dist = new_dist
                closest = node
        return closest, old_dist
    else:
        return None, None

def find_closest_edge(point, edges):
    if len(edges) > 0:
        closest = edges[0]
        old_dist = distance_to_edge(point, closest)
        for edge in edges:
            new_dist=distance_to_edge(point, edge)
            if new_dist<old_dist:
                old_dist = new_dist
                closest = edge
        return closest, old_dist
    else:
        return None, None
        
def draw_nodes(img, nodes, node_color=(255,0,0), special_nodes=(), special_node_colors=(), size=3, thickness=2):
    frame = np.copy(img)
    for node in nodes:
        if special_nodes != None and node in special_nodes:
            frame = cv2.circle(frame, node, size, special_node_colors[special_nodes.index(node)], thickness=thickness)
        else:
            frame = cv2.circle(frame, node, size, node_color, thickness=thickness)
    return frame

def draw_edges(img, edges, edge_color=(0,255,0), special_edges=(), special_edge_colors=(), thickness = 3, use_arrows = True):
    frame = np.copy(img)
    for edge in edges:
        if special_edges == None or not edge in special_edges:
            if use_arrows:
                frame = cv2.arrowedLine(frame, edge[0], edge[1], edge_color, thickness=thickness, line_type=cv2.LINE_AA)
            else:
                frame = cv2.line(frame, edge[0], edge[1], edge_color, thickness=thickness, lineType=cv2.LINE_AA)

    for edge in edges:
        if special_edges != None and edge in special_edges:
            if use_arrows:
                frame = cv2.arrowedLine(frame, edge[0], edge[1], special_edge_colors[special_edges.index(edge)], thickness=thickness, line_type=cv2.LINE_AA)
            else:
                frame = cv2.line(frame, edge[0], edge[1], special_edge_colors[special_edges.index(edge)], thickness=thickness, lineType=cv2.LINE_AA)

    
    return frame

def distance_to_edge(point, edge): # x3,y3 is the point
    ((x1, y1), (x2, y2)) = edge
    x3, y3 = point
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx*dx + dy*dy)**.5

    return dist

def localise_in_graph(point, edges):
    edge_closest, _ = find_closest_edge(point, edges)
    if edge_closest == None:
        return (), []
    else:
        ret_edges = []
        for edge in edges:
            if edge[0] == edge_closest[1]:
                ret_edges.append(edge)
        return edge_closest, ret_edges


def delete_node(node, nodes, edges):
    edges = [edge for edge in edges if edge[0] != node and edge[1] != node]
    nodes.remove(node)
    return nodes, edges
def newMod(a,b):
    res = a%b
    return res if not res else res-b if a<0 else res
def click(event, x, y, flags, param):
    global nodes
    global edges
    global new_edge_first_point
    global closest_edge 
    global closest_node
    global resizable_image
    global mouse_coords
    global intersection
    point = resizable_image.out_to_in_point((x,y))
    mouse_coords = point
    if state == 0:
        if event == click_down:
            nodes.append(point)
        elif event == click_up:
            pass

    elif state == 1:
        if new_edge_first_point == None:
            closest_node, _ = find_closest_node(point, nodes)
            if event == click_down:
                pass
            elif event == click_up:
                new_edge_first_point = closest_node
        else:
            closest_node, _ = find_closest_node(point, nodes)
            if event == click_down:
                pass
            elif event == click_up:
                if new_edge_first_point != closest_node:
                    edges.append((new_edge_first_point, closest_node))
                    new_edge_first_point = None
                else:
                    new_edge_first_point = None


    elif state == 2:
        closest_node, dist_node = find_closest_node(point, nodes)
        closest_edge, dist_edge = find_closest_edge(point, edges)
        try:
            if dist_edge >= dist_node:
                closest_edge = None
            else:
                closest_node = None
        except TypeError:
            pass
        if event == click_down:
            pass
        elif event == click_up:
            if closest_node != None:
                modes, edges = delete_node(closest_node, nodes, edges)
            elif closest_edge != None:
                edges.remove(closest_edge)
    
    elif state == 6:
        closest_node, _ = find_closest_node(point, nodes)
        if event == click_up:
            intersection.append(closest_node)
            if len(intersection) == 8:
                for i in range(0, len(intersection), 2):
                    for j in range(1,len(intersection)+1, 2):
                        if newMod(j+1,8) != i or u_turn_at_intersections:
                            edges.append((intersection[i], intersection[j]))
                intersection = []

if __name__ == "__main__":
    print(7+1%8)
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, click)
    resizable_image = ResizableImage(win_scale)
    if graph_load_dir != "":
        try:
            with open(graph_load_dir, "rb") as file:
                nodes, edges = pickle.load(file)
        except FileNotFoundError:
            pass
    while True:
        frame = np.copy(image)
        if state == 0:
            frame = draw_edges(frame, edges)
            frame = draw_nodes(frame, nodes)
        elif state == 1:
            frame = draw_edges(frame, edges)
            frame = draw_nodes(frame, nodes, special_nodes=[closest_node, new_edge_first_point], special_node_colors=[(0,255,255), (255,255,0)])
        elif state == 2:
            if closest_edge != None:
                frame = draw_edges(frame, edges, special_edges=[closest_edge], special_edge_colors=[(0,0,255)])
            else:
                frame = draw_edges(frame, edges)
                
            if closest_node != None:
                frame = draw_nodes(frame, nodes, special_nodes=[closest_node], special_node_colors=[(0,0,255)])
            else:
                frame = draw_nodes(frame, nodes)
        
        elif state == 3:
            frame = draw_edges(frame, edges)
            frame = draw_nodes(frame, nodes)
            height, width, depth = np.shape(frame)
            frame = cv2.putText(frame, "Quit?",(width//2-200, height//2), cv2.FONT_HERSHEY_COMPLEX, 5, (0,0,255), thickness=10, lineType=cv2.LINE_AA)
        elif state == 4:
            frame = draw_edges(frame, edges)
            frame = draw_nodes(frame, nodes)
            height, width, depth = np.shape(frame)
            frame = cv2.putText(frame, "Clear?",(width//2-250, height//2), cv2.FONT_HERSHEY_COMPLEX, 5, (0,0,255), thickness=10, lineType=cv2.LINE_AA)
        elif state == 5:
            current_edge, next_edges = localise_in_graph(mouse_coords, edges)
            print(f"\n\nAvailable roads: {len(next_edges)}")
            print("Possible nodes to travel to")
            for edge in next_edges:
                print(edge[1])
            frame = draw_edges(frame, edges, special_edges=[current_edge]+next_edges, special_edge_colors=[(0,255,255)]+[(255,0,255) for i in range(len(next_edges))])
            frame = draw_nodes(frame, nodes)
        elif state == 6:
            frame = draw_edges(frame, edges)
            frame = draw_nodes(frame, nodes, special_nodes=[closest_node]+ intersection, special_node_colors=[(255,0,255)]+[(255,255,0) if i%2==0 else (0,255,255) for i in range(len(intersection))])


        if state != 6:
            intersection = []
        
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
                nodes = []
                edges = []
                state = 0
            else:
                state = 4
        elif key == ord(localise_mode):
            state = 5
        elif key == ord(intersection_mode):
            state=6
        elif key == -1:
            continue
        else:
            state = 0

    cv2.destroyAllWindows()
    if graph_save_dir != "":
        with open(graph_save_dir,'wb') as file:
            pickle.dump((nodes, edges), file)
import sys
sys.path.insert(1, '/home/batu/projects/self-driving-taxi/catkin_ws/src/panorama/scripts/python-utils')
from utils import *
import numpy as np
import cv2
import math as m


class Node():    
    def __init__(self, position):
        self.travelable = False
        position = np.array(position, dtype=int)
        assert np.shape(position) == (2,1), f"Incorrect input shape {np.shape(position)} , expected (2,1)"
        self.position = position

    def is_travelable(self):
        return self.travelable

    def distance_to(self, point):
        point = np.array(point, dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(point)} , expected (2,1)"
        return m.sqrt((self.position[0][0] -point[0][0])**2+(self.position[1][0] -point[1][0])**2)
    
    def draw(self, image, color=(255,0,0,), size=4): 
        node = (self.position[0][0], self.position[1][0])
        return cv2.circle(image, node, int(m.ceil(size/2)), color, thickness=size, lineType=cv2.LINE_AA)

    def __str__(self):
        return str(self.position)

    # @classmethod
    # def fromSameObject(cls, node):
    #     assert type(node) == cls, f"node must be of type {cls}, not {type(node)}"
    #     return cls(np.copy(node.position))

class Edge():
    def __init__(self, node1, node2):
        assert type(node1) == Node, f"node1 must be a node, not a {type(node1)}"
        assert type(node2) == Node, f"node2 must be a node, not a {type(node2)}"
        self.nodes = (node1, node2)
        self.travelable=True
    def is_travelable(self):
        return self.travelable
   
    def distance_to(self, point):
        point = np.array(point,dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(point)} , expected (2,1)"
        node1, node2 = self.nodes
        x1 = node1.position[0][0]
        y1 = node1.position[1][0]
        x2 = node2.position[0][0]
        y2 = node2.position[1][0]
        x3 = point[0][0]
        y3 = point[1][0]
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

    def draw(self, image, color=(0,255,0), size=3, is_directed=True):
        x1 = self.nodes[0].position[0][0]
        y1 = self.nodes[0].position[1][0]
        x2 = self.nodes[1].position[0][0]
        y2 = self.nodes[1].position[1][0]
        if is_directed:
            return cv2.arrowedLine(image, (x1, y1), (x2,y2), color, thickness=size, line_type=cv2.LINE_AA)
        else:
            return cv2.line(image, (x1, y1), (x2,y2), color, thickness=size, lineType=cv2.LINE_AA)
    def entering_nodes(self):
        return [self.nodes[0]]
    def exiting_nodes(self):
        return [self.nodes[1]]
    def travelable_nodes(self, invert_direction=False, undirected=False):
        entering_node = self.nodes[0]
        exiting_node = self.nodes[1]
        if undirected:
            legal_travelable_nodes = [exiting_node, entering_node]
            illegal_travelable_nodes = []
            return legal_travelable_nodes, illegal_travelable_nodes
        else:
            if invert_direction == False:
                legal_travelable_nodes = [exiting_node]
                illegal_travelable_nodes = [entering_node]
                return legal_travelable_nodes, illegal_travelable_nodes
            else:
                legal_travelable_nodes = [entering_node]
                illegal_travelable_nodes = [exiting_node]
                return legal_travelable_nodes, illegal_travelable_nodes

    def contains_node(self, node):
        assert type(node) == Node, "node must be of Node type"
        return node in self.nodes

    # @classmethod
    # def fromSameObject(cls, edge):
    #     assert type(edge) == cls, f"edge must be of type {cls}, not {type(edge)}"
    #     return cls(Node.fromSameObject(edge.nodes[0]), Node.fromSameObject(edge.nodes[1]))

class Intersection():

    def __init__(self, nodes):
        assert len(nodes)%2 == 0 and len(nodes)>=4, f"Length of nodes must be >= 4 and be a multiple of 2"
        for node in nodes:
            assert type(node) == Node, "All elements of nodes must be of type node"
        self.nodes = nodes
        self.travelable = True #can it be traveled on?

    def is_travelable(self):
        return self.travelable
    def entering_nodes(self):
        return [self.nodes[i] for i in range(0,len(self.nodes),2)]
    def exiting_nodes(self):
        return [self.nodes[i] for i in range(1,len(self.nodes)+1,2)]
    
    def __center_point(self):
        X = 0
        Y = 0
        for node in self.nodes:
            X+=node.position[0][0]
            Y+=node.position[1][0]
        X/=len(self.nodes)
        Y/=len(self.nodes)
        return np.array([[X],[Y]], dtype=int)

    def draw(self, image, color=(0,255,0), size=3):
        ret = np.copy(image)
        if color==(0,255,0):
            text_color = (255,0,255)
            line_color = (0,255,0)
        else:
            text_color=color
            line_color=color
        for i in range(len(self.nodes)):
            edge = Edge(self.nodes[i], self.nodes[(i+1)%len(self.nodes)])
            ret = edge.draw(ret, line_color, size, False)
        center = self.__center_point()
        x = center[0][0]
        y = center[1][0]
        ret = putTextCenter(ret, "I", (x, y), cv2.FONT_HERSHEY_COMPLEX, size, text_color,thickness=size)
        # ret = cv2.putText(ret, "I", (x, y), cv2.FONT_HERSHEY_COMPLEX, size, text_color,thickness=size)
        return ret
    def distance_to(self, point):
        # self_center_node = Node(self.__center_point())
        edges = [Edge(self.nodes[i], self.nodes[(i+1)%len(self.nodes)]) for i in range(len(self.nodes))]
        min_dist = m.inf
        for edge in edges:
            dist = edge.distance_to(point)
            if min_dist > dist:
                min_dist = dist
        return min_dist

    def travelable_nodes(self, invert_direction=False, undirected=False):
        entering_nodes = self.entering_nodes()
        exiting_nodes = self.exiting_nodes()
        if undirected:
            legal_travelable_nodes = exiting_nodes + entering_nodes
            illegal_travelable_nodes = []
            return legal_travelable_nodes, illegal_travelable_nodes
        else:
            if invert_direction == False:
                legal_travelable_nodes = exiting_nodes
                illegal_travelable_nodes = entering_nodes
                return legal_travelable_nodes, illegal_travelable_nodes
            else:
                legal_travelable_nodes = entering_nodes
                illegal_travelable_nodes = exiting_nodes
                return legal_travelable_nodes, illegal_travelable_nodes
    def contains_node(self, node):
        assert type(node) == Node, "node must be of Node type"
        return node in self.nodes

    # @classmethod
    # def fromSameObject(cls, intersection):
    #     assert type(intersection) == cls, f"intersection must be of type {cls}, not {type(intersection)}"
    #     return cls([Node.fromSameObject(node) for node in intersection.nodes])

class Graph():
    def __init__(self):
        self.elements = []
    def draw(self, image, special_elements=[], special_element_colors=[], draw_order=[]):
        ret = np.copy(image)
        drawn_elements = [False for i in range(len(self.elements))]
        for element_type in draw_order:
            for i, element in enumerate(self.elements):
                if not element in special_elements and type(element)==element_type:
                    ret = element.draw(ret)
                    drawn_elements[i] = True
        for i, element in enumerate(self.elements):
                if not element in special_elements and drawn_elements[i]==False:
                    ret = element.draw(ret)
                    drawn_elements[i] = True
        for element in self.elements:
            if element in special_elements:
                ret = element.draw(ret, color=special_element_colors[special_elements.index(element)])
        return ret

    def add_element(self, element):
        self.elements.append(element)
    
    def remove_element(self, element):
        if element in self.elements:
            self.elements.remove(element)
            return True
        else:
            return False
    
    def find_closest_element(self, point, element_type=None, return_distance=False, strictly_travelable=False):
        point = np.array(point, dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(position)} , expected (2,1)"
        assert type(return_distance) == bool, f"return_distance type must be bool, not {type(return_distance)}"
        closest_element = None
        closest_element_distance = m.inf
        for element in self.elements:
            if (element_type==None or type(element)==element_type) and (strictly_travelable <= element.is_travelable()):
                current_distance = element.distance_to(point)
                if current_distance<closest_element_distance:
                    closest_element = element
                    closest_element_distance=current_distance
        if return_distance:
            return closest_element, closest_element_distance
        else:
            return closest_element

    # def find_travelable_nodes(self, point, invert_direction=False, undirected=False):
    #     point = np.array(point, dtype=int)
    #     assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(position)} , expected (2,1)"
    #     current_element = self.find_closest_element(point, strictly_travelable=True)
    #     if current_element == None:
    #         return [], []
    #     legal_nodes, illegal_nodes = current_element.travelable_nodes(invert_direction, undirected)
    #     return legal_nodes, illegal_nodes

    def clear(self):
        self.elements = []

    # @classmethod
    # def fromSameObject(cls, graph):
    #     assert type(graph) == cls, f"graph must be of type {cls}, not {type(graph)}"
    #     ret = cls()
    #     for element in graph.elements:
    #         ret.add_element(type(element).fromSameObject(element))
    #     return ret
    
def localise_in_graph(graph, point):
    current_travelable_element = graph.find_closest_element(point, strictly_travelable=True)

    legal_nodes, illegal_nodes = current_travelable_element.travelable_nodes(False, False)

    # print(f"\nLegal node count: {len(legal_nodes)}")
    # print("Possible legal nodes to travel to")
    # for node in legal_nodes:
    #     print(node.position)
    
    # special_elements = [current_travelable_element]+legal_nodes
    # special_element_colors = [(0,255,255)]+[(255,0,255) for i in range(len(special_elements))]

    next_travelable_elements = []
    for node in legal_nodes:
        for element in graph.elements:
            if element.travelable == True and node in element.entering_nodes():
                next_travelable_elements.append(element)
                # special_elements.append(element)
                # special_element_colors.append((255,0,255))
    
    return current_travelable_element, next_travelable_elements
    
if __name__=="__main__":
    node = Node(np.array([[0,0]]).T)
    print(type(node).fromSameObject(node))
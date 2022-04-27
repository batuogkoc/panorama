import numpy as np
import cv2

class Node():
    def __init__(self):
        self.position = np.zeros((2,1))
        self.__travelable = False #can it be traveled on?
    def travelable(self):
        return self.__travelable
    def __init__(self, position):
        position = np.array(position, dtype=int)
        assert np.shape(position) == (2,1), f"Incorrect input shape {np.shape(position)} , expected (2,1)"
        self.position = position
    def distance_to(self, point):
        point = np.array(point, dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(point)} , expected (2,1)"
        return m.sqrt(np.dot(self.position, point))
    def draw(self, image, color=(255,0,0,), size=4): 
        return cv2.circle(image, node, int(m.ceil(size/2)), color, thickness=size, lineType=cv2.LINE_AA)

class Edge():
    def __init__(self):
        self.nodes = (Node(),Node())
        self.__travelable = True #can it be traveled on?

    def travelable(self):
        return self.__travelable
    def __init__(self, node1, node2):
        assert type(node1) == Node, f"node1 must be a node, not a {type(node1)}"
        assert type(node2) == Node, f"node2 must be a node, not a {type(node2)}"
        self.nodes = (node1, node2)
    def distance_to(self, point):
        point = np.array(point,dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(point)} , expected (2,1)"
        node1, node2 = self.nodes
        x1 = node1.position[0][0]
        y1 = node1.position[0][1]
        x2 = node1.position[0][0]
        y2 = node1.position[0][1]
        x3 = point[0][0]
        y3 = point[0][1]
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

    def draw(self, image, color=(0,255,255), size=3, is_directed=True):
        x1 = node1.position[0][0]
        y1 = node1.position[0][1]
        x2 = node1.position[0][0]
        y2 = node1.position[0][1]
        if is_directed:
            return cv2.arrowedLine(image, (x1, y1), (x2,y2), edge_color, thickness=size, line_type=cv2.LINE_AA)
        else:
            return cv2.line(image, (x1, y1), (x2,y2), edge_color, thickness=size, lineType=cv2.LINE_AA)

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

class Intersection():
    def __init__(self, nodes):
        assert len(nodes)%2 == 0 and len(nodes)>=4, f"Length of nodes must be >= 4 and be a multiple of 2"
        for node in nodes:
            assert type(node) == Node, "All elements of nodes must be of type node"
        self.nodes = nodes
        self.__travelable = True #can it be traveled on?
    def travelable(self):
        return self.__travelable
    def entering_nodes(self):
        return [self.nodes[i] for i in range(0,len(self.nodes),2)]
    def exiting_nodes(self):
        return [self.nodes[i] for i in range(1,len(self.nodes)+1,2)]
    
    def __center_point(self):
        X = 0
        Y = 0
        for node in self.nodes:
            X+=node.position[0][0]
            Y+=node.position[0][1]
        X/=len(self.nodes)
        Y/=len(self.nodes)
        return np.arrray([[X],[Y]])

    def draw(self, image, color=(0,255,0), size=3):
        ret = np.copy(image)
        for i in range(len(self.nodes)):
            edge = Edge(self.nodes[i], self.nodes[(i+1)%len(self.nodes)])
            ret = edge.draw(ret, color, size, False)
        center = self.__center_point()
        x = center[0][0]
        y = center[0][1]
        ret = cv2.putText(ret, "I", (x,y), cv2.FONT_HERSHEY_SIMPLEX, size*10, color)
        return ret
    def distance_to(self, point):
        self_center_node = Node(self.__center_point())
        return self_center_node.distance_to(point)

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

class Graph():
    def __init__(self):
        self.elements = []
    def draw(self, image, special_elements=[], special_element_colors=[]):
        ret = np.copy(image)
        for element in elements:
            if not element in special_elements:
                ret = element.draw(ret)
        for element in elements:
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
    
    def find_closest_element(self, point, return_distance=False):
        point = np.array(point, dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(position)} , expected (2,1)"
        assert type(return_distance) == bool, f"return_distance type must be bool, not {type(return_distance)}"
        closest_element = self.elements[0]
        closest_element_distance = closest_element.distance_to(point)
        for element in self.elements:
            current_distance = element.distance_to(point)
            if current_distance<closest_element_distance:
                closest_element = element
                closest_element_distance=element
        if return_distance:
            return closest_element, closest_element_distance
        else:
            return closest
    
    def find_travelable_nodes(self, point, invert_direction=False, undirected=False):
        point = np.array(point, dtype=int)
        assert np.shape(point) == (2,1), f"Incorrect point shape {np.shape(position)} , expected (2,1)"
        current_element = self.find_closest_element(point)
        legal_nodes, illegal_nodes = current_element.travelable_nodes(invert_direction, undirected)
        return legal_nodes, illegal_nodes


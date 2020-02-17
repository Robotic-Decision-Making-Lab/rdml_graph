# Node.py
# Written Ian Rankin October 2019 - edited Feb 2020
#
# A generic node structure for a graph. Can be extended to include more
# information about the node.

from rdml_graph import State
from rdml_graph import Edge

class Node(State):
    # constructor
    # @param id - the integer the Node repersents.
    def __init__(self, id):
        self.e = []
        self.id = id

    def successor(self):
        return [(edge.c, edge.getCost()) for edge in self.e]

    def addEdge(self, edge):
        self.e.append(edge)

    def getEdges(self):
        return self.e

    ############### operator overloading

    # == operator
    # Only looks at the id's to check if it is the same node.
    def __eq__(self, other):
        return self.id == other.id

    # != operator
    # returns inverse of equals sign.
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)

    # str(self) operator
    # Returns a quick human readable string
    def __str__(self):
        result = 'node(id='+ str(id) + ', edges={'
        for edge in e:
            result += str(edge)+','
        result += '})'
        return result


class GeometricNode(Node):
    # Constructor
    # @param id - the integer that is the 'index' of the node.
    # @param pt - numpy array repersenting spatial point.
    def __init__(self, id, pt):
        super(GeometricNode, self).__init__(id)
        self.pt = pt

    # str(self) operator
    # Returns a quick human readable string
    def __str__(self):
        result = 'node(id='+ str(id) + ', pt='+ str(self.pt) +' edges={'
        for edge in e:
            result += str(edge)+','
        result += '})'
        return result

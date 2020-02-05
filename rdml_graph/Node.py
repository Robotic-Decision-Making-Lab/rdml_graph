# Node.py
# Written Ian Rankin October 2019 - edited Feb 2020
#
# A generic node structure for a graph. Can be extended to include more
# information about the node.

import State
import Edge

class Node(State.State):
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

class GeometricNode(Node):
    # Constructor
    # @param id - the integer that is the 'index' of the node.
    # @param pt - numpy array repersenting spatial point.
    def __init__(self, id, pt):
        super(GeometricNode, self).__init__(id)
        self.pt = pt

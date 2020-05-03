# HomotopyNode.py
# Written Ian Rankin - February 2020
#
# Homotopy augumented node.
# This is a set of nodes built on-top of a standard
# Node and graph, but has a different successor function
# Based on work by following paper:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
#

from ..core import State
from ..core import Node
from ..core import Edge
from . import HSignature

import numpy as np

import pdb

# getWaypoints
# get waypoints from a list of HomotopyNodes.
# @param path - a list of homotopy nodes.
#
# @return 2d numpy array of waypoints, (n x 2)
def getWaypointsHomotopy(path):
    pts = np.empty((len(path), 2))
    for i, homotopy in enumerate(path):
        pts[i] = homotopy.node.pt

    return pts

class HomotopyNode(State):
    # constructor
    # @param node - the input Node for homotopy graph.
    # @param h_sign - the input H signature.
    # @param parent - [optional] the edge from the parent HomotopyNode
    # @param root - [optional] the root node of the homotopy graph.
    def __init__(self, n, h_sign, parentEdge=None, root=None):
        self.node = n
        self.h_sign = h_sign
        self.parentEdge = parentEdge
        self.root = root

    # successor function for Homotopy node.
    def successor(self):
        result = []
        for edge in self.node.e:
            newHSign = self.h_sign.copy()
            goodHSign = newHSign.edgeCross(edge)

            if goodHSign:
                succ = HomotopyNode(n=edge.c, h_sign=newHSign,\
                                parentEdge=edge, root=self.root)
                result += [(succ, edge.getCost())]
        #pdb.set_trace()
        return result

    ################## operator overloading

    # ==
    # equals checks other is the same class and then checks node and h-signature
    # for equality.
    def __eq__(self, other):
        if not isinstance(other, HomotopyNode):
            return False
        return self.node == other.node and self.h_sign == other.h_sign and \
                (self.root is None or other.root is None or self.root == other.root)

    # !=
    # opposite of equals
    def __ne__(self, other):
        return not self == other

    # str()
    # prints out info about the currnet Homotopy Node
    def __str__(self):
        return 'HomotopyNode(h-sign='+ str(self.h_sign) +', n=' + str(self.node) + ')'

    # hash function overload
    # This hash takes into account both the node hash (should be defined),
    # and the h signatures hash (also defined).
    # parent edge is not considered.
    ###### THIS is actually important for SEARCHES as it defines what is considered
    # already explored.
    def __hash__(self):
        return hash((self.node, self.h_sign, self.root))

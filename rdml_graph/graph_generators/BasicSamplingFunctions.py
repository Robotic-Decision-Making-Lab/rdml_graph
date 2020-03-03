# BasicSamplingFunctions.py
# Written Ian Rankin March 2020
#
#

import numpy as np

from ..core import GeometricNode
from ..core import Edge
from ..homotopy import HomotopyEdge

########################## Sampling functions for PRM's

def sample2DUniform(map, num_samples):
    samples = np.random.random((num_samples, 2))  * map['size']
    nodes = [GeometricNode(i, samples[i]) for i in range(samples.shape[0])]
    return nodes, samples



########################## Collision checking algorithms

# no collision
# a simple function to always indicate there are no collisions for the PRM
# to grow.
# @param u - one of the input nodes.
# @param v - the second input node.
# @param map - the input map to check for collisions using.
def noCollision(u , v, map):
    return False


######################### Edge connection functions

# EdgeConnection
# Creates a connection between node u to node v.
# Default version just connects the two using an Edge object
# @param parent - parent node of connection
# @param child - child node of connection.
# @param map - the input map (not needed for this connection)
# @param cost - the cost of the connection (if None assume it must caluclate the cost)
#
# @return - cost of edge.
def EdgeConnection(parent, child, map, cost = None):
    if cost is None:
        cost = np.linalg.norm(parent.pt - child.pt, ord=2)
    parent.addEdge(Edge(parent, child, cost))
    return cost


# HomotopyEdgeConn
# Creates a connection between parent and child node using a Homotopy Edge
# @param parent - parent node of connection
# @param child - child node of connection.
# @param map - the input map MUST have map['features'] defined as a 2d numpy array
# @param cost - the cost of the connection (if None assume it must caluclate the cost)
#
# @return - cost of edge.
def HomotopyEdgeConn(parent, child, map, cost = None):
    if cost is None:
        cost = np.linalg.norm(parent.pt - child.pt, ord=2)
    parent.addEdge(HomotopyEdge(parent, child, cost=cost, features=map['features']))
    return cost

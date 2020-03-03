# PRM.py
# Written Ian Rankin - February 2020
#
# A set of functions that generate a Probabilistic RoadMap (PRM).
# A very generic version of the PRM is used to generate all other
# types of PRM's.

from ..core import GeometricNode
from ..core import Edge
import numpy as np
import scipy.spatial as spa

def sample2DUniform(map, num_samples):
    samples = np.random.random((num_samples, 2))  * map['size']
    nodes = [GeometricNode(i, samples[i]) for i in range(samples.shape[0])]
    return nodes, samples

# no collision
# a simple function to always indicate there are no collisions for the PRM
# to grow.
# @param u - one of the input nodes.
# @param v - the second input node.
# @param map - the input map to check for collisions using.
def noCollision(u , v, map):
    return False

# EdgeConnection
# Creates a connection between node u to node v.
# Default version just connects the two using an Edge object
# @param u - parent
def EdgeConnection(parent, child, map, cost = None):
    if cost is None:
        cost = np.linalg.norm(parent.pt - child.pt, ord=2)
    parent.addEdge(Edge(parent, child, cost))
    return cost



# PRM
# Generates a Probabilistic RoadMap (PRM) of the sample space.
# The exact sample space can be determined by setting different sample functions,
# collision functions, and connection functions.
# @param map - a dictionary of map values (by default should have map['size'] defined)
# @param num_points - the number of points to generate using the PRM
# @param r - the radius to check connections between.
# @param sampleF - the sampling function for the PRM
#               sampleF(map, num_samples)
# @param collision - the collision function to check for connection between nodes:
#               collision(parent, child, map) parent->child nodes, map is the given map of the PRM.
# @param connection - the connection function setting costs and exact method to calculate.
#               connection(parent, child, map, cost=None) - connects the parent to the child node.
# @param bidirectional - sets if the PRM is guarenteed to be bidirectional and
#                if bidirectional collisions and costs need to be checked.
#
# @return - list of nodes
def PRM(map, num_points, r, sampleF=sample2DUniform, collision=noCollision, \
        connection=EdgeConnection, bidirectional=True):
    nodes, pts = sampleF(map, num_points)

    # connect points using kdtree for graph.
    # generate kd-tree from data.
    nn = spa.cKDTree(pts) # nn = nearest neighbors search.


    # go through every point and check for connections.
    for i in range(len(nodes)):
        n = nodes[i] # current node.

        # uses 2-norm for measuring nearby points.
        nearPtIdxs = nn.query_ball_point(pts[i], r) # get indcies of nearby points

        for idx in nearPtIdxs:
            # check that edge hasn't already been checked.
            # by ignoring all points already having connections.
            if (not bidirectional) or idx > i:
                if not collision(n, nodes[idx], map):
                    # connect the two nodes with a cost function determined by the edge connection.
                    cost = EdgeConnection(n, nodes[idx], map)
                    if bidirectional:
                        EdgeConnection(nodes[idx], n, map, cost) # set other direction of PRM.
    return nodes

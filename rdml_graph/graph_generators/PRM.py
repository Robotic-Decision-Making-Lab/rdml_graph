# Copyright 2020 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# PRM.py
# Written Ian Rankin - February 2020
#
# A set of functions that generate a Probabilistic RoadMap (PRM).
# A very generic version of the PRM is used to generate all other
# types of PRM's.

from ..core import GeometricNode
from ..core import Edge
import scipy.spatial as spa

import numpy as np
from .BasicSamplingFunctions import sample2DUniform, noCollision, EdgeConnection

import pdb

# PRM
# Generates a Probabilistic RoadMap (PRM) of the sample space.
# The exact sample space can be determined by setting different sample functions,
# collision functions, and connection functions.
# @param map - a dictionary of map values (by default should have map['size'] defined)
# @param num_points - the number of points to generate using the PRM
# @param r - the radius to check connections between.
# @param initialNodes - a list of initial nodes that need to be added to the PRM
#                       added at the front of the PRM map. (Assumes nodes have .pt variable)
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
def PRM(map, num_points, r, initialNodes=[], sampleF=sample2DUniform, \
        collision=noCollision, connection=EdgeConnection, bidirectional=True):

    maxId = -1
    for n in initialNodes:
        if n.id > maxId:
            maxId = n.id

    #pdb.set_trace()
    maxId += 1

    # sample all points
    nodes, pts = sampleF(map, num_points, idStart=maxId)

    if len(initialNodes) > 0:
        initPts =  np.empty((len(initialNodes),) + initialNodes[0].pt.shape)
        for i in range(len(initialNodes)):
            initPts[i] = initialNodes[i].pt

        pts = np.append(initPts, pts, axis=0)
        nodes = initialNodes + nodes

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
                    cost = connection(n, nodes[idx], map)
                    if bidirectional:
                        connection(nodes[idx], n, map, cost) # set other direction of PRM.
    return nodes

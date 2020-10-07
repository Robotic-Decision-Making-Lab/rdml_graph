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
# BasicSamplingFunctions.py
# Written Ian Rankin March 2020
#
#

import numpy as np

from ..core import GeometricNode
from ..core import Edge
from ..homotopy import HEdge
from ..homotopy import HomologySignature

########################## Sampling functions for PRM's

def sample2DUniform(map, num_samples, idStart=0):
    height = map['height']
    width = map['width']

    samples = np.random.random((num_samples, 2))  * np.array([width, height]) - np.array([width/2, height/2])
    nodes = [GeometricNode(i + idStart, samples[i]) for i in range(samples.shape[0])]
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


# HEdgeConn
# Creates a connection between parent and child node using a Homotopy Edge
# @param parent - parent node of connection
# @param child - child node of connection.
# @param map - the input map MUST have map['features'] defined as a 2d numpy array
# @param cost - the cost of the connection (if None assume it must caluclate the cost)
#
# @return - cost of edge.
def HEdgeConn(parent, child, map, cost = None):
    if cost is None:
        cost = np.linalg.norm(parent.pt - child.pt, ord=2)

    # if 'ray_angle' in map:
    #     ray_angle = map['ray_angle']
    # else:
    #     ray_angle = np.pi/2
    # if 'h_type' in map and map['h_type'].lower() == 'homotopy':
    #     h_sign =

    h_sign = HomologySignature(map['hazards'].shape[0])
    if 'ray_angle' in map:
        parent.addEdge(HEdge(parent, child, h_sign, cost=cost, \
                        features=map['hazards'], ray_angle=map['ray_angle']))
    else:
        parent.addEdge(HEdge(parent, child, h_sign, cost=cost,\
                            features=map['hazards']))

    return cost

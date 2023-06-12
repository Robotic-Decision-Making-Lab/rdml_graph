# Copyright 2023 Ian Rankin
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
# ConnectedGrid.py
# Written Ian Rankin - June 2023
#
# Generate a graph object for a connected grid of points.

from ..core import GeometricNode
import shapely.geometry as geo

import numpy as np
from .BasicSamplingFunctions import sample2DUniform, noCollision, EdgeConnection

import pdb


## connected_grid
# Generate a graph as a connected 4 or 8 grid.
# Can define size of the grid given max sizes and ticks of the underlying field.
# @param map - map with options {'bounding', 'obs', 'hazards'}
# @param x_ticks - array like of N ticks
# @param y_ticks - array like of M ticks
# @param sampleF - the sampling function for the PRM
#               sampleF(map, num_samples)
# @param collision - the collision function to check for connection between nodes:
#               collision(parent, child, map) parent->child nodes, map is the given map of the PRM.
# @param connection - the connection function setting costs and exact method to calculate.
#               connection(parent, child, map, cost=None) - connects the parent to the child node.
# @param grid_size - [opt] the size of the grid given the x and y ticks.
# @param conn_8 - [opt] true 8-connected grid (diagonals), false, 4-connected grid
# @param bidirectional - [opt] sets if the PRM is guarenteed to be bidirectional and
#                if bidirectional collisions and costs need to be checked.
#
# @return a list of nodes with grid conencted edges
def connected_grid(map, x_ticks, y_ticks, collision=noCollision, connection=EdgeConnection, grid_size=1, conn_8=True, bidirectional=True):
    id_num = 0
    G = []

    # generate all needed nodes
    for i in range(0, len(x_ticks), grid_size):
        for j in range(0, len(y_ticks), grid_size):
            pt = np.array([x_ticks[i], y_ticks[j]])

            n = GeometricNode(id_num, pt)
            G.append(n)
            id_num += 1

    # generate edge connections
    x_size = int(len(x_ticks) / grid_size)
    y_size = int(len(y_ticks) / grid_size)

    for i in range(0, x_size):
        for j in range(0, y_size):
            n = G[calc_index(i,j,x_size, y_size)]

            # right edge connection
            if i != (x_size-1):
                edge_n = G[calc_index(i+1, j, x_size, y_size)]          
                if not collision(n, edge_n, map):
                    connection(n, edge_n, map)
                    if bidirectional:
                        connection(edge_n, n, map)
                if not bidirectional:
                    if not collision(n, edge_n, map):
                        connection(edge_n, n, map)

            # Down edge connection
            if j != (y_size-1):
                edge_n = G[calc_index(i, j+1, x_size, y_size)]          
                if not collision(n, edge_n, map):
                    connection(n, edge_n, map)
                    if bidirectional:
                        connection(edge_n, n, map)
                if not bidirectional:
                    if not collision(n, edge_n, map):
                        connection(edge_n, n, map)

            # diagonal down-right edge connection
            if conn_8 and j != (y_size-1) and i != (x_size-1):
                edge_n = G[calc_index(i+1, j+1, x_size, y_size)]          
                if not collision(n, edge_n, map):
                    connection(n, edge_n, map)
                    if bidirectional:
                        connection(edge_n, n, map)
                if not bidirectional:
                    if not collision(n, edge_n, map):
                        connection(edge_n, n, map)

            # diagonal down-left edge connection
            if conn_8 and j != (y_size-1) and i != 0:
                edge_n = G[calc_index(i-1, j+1, x_size, y_size)]          
                if not collision(n, edge_n, map):
                    connection(n, edge_n, map)
                    if bidirectional:
                        connection(edge_n, n, map)
                if not bidirectional:
                    if not collision(n, edge_n, map):
                        connection(edge_n, n, map)
        # end for loop j
    # end for loop i 

    return G
# end connected_grid


def calc_index(x,y, x_size, y_size):
    return x*y_size + y


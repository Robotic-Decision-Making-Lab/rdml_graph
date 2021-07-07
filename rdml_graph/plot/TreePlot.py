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
## TreePlot.py
# Written Ian Rankin - February 2020
#
# A set of functions to plot a tree from a certain node.
# Can also be used to show the tree structure generated by a BFS.

import numpy as np
import matplotlib.pyplot as plt
from rdml_graph.core import Node
from collections import deque

## plotTree
# plot's a tree using a BFS search through the environment.
# Each level is given equal space.
# @param root - the top of the tree given as a Node
# @param max_levels - the max number of levels of the tree to plot.
# @param show_labels - shows the labels the nodes using the id of node.
def plotTree(root, max_levels=-1, show_labels=False):
    frontier = deque()
    frontier.append((root, 0))
    explored = set()

    nodesLevels = [[]]

    while len(frontier) > 0:
        n, level = frontier.pop()
        if max_levels != -1 and level >= max_levels:
            break

        if n not in explored:
            explored.add(n)
            successors = n.successor()

            # add the current node to node levels.
            if len(nodesLevels) > level:
                nodesLevels[level].append(n)
            else:
                #add a new layer
                nodesLevels.append([n])

            for succ, cost in successors:
                if succ not in explored:
                    frontier.append((succ, level+1))

    # BFS search done, plot tree
    num_levels = len(nodesLevels)
    total_nodes = sum([len(lev) for lev in nodesLevels])

    pts = np.empty((total_nodes, 2))
    edges = np.empty((0,2))

    nodesToIdx = {}
    idx = 0

    # Generate node locations
    for l in range(num_levels):
        level = nodesLevels[l]
        for i in range(len(level)):
            nodesToIdx[level[i]] = idx
            pts[idx][1] = float(-l)
            pts[idx][0] = float(i)

            idx += 1

    # Generate edges
    for l in range(num_levels):
        level = nodesLevels[l]
        for i in range(len(level)):
            n = level[i]
            idx = nodesToIdx[n]

            # Go through each child node.
            for e in n.e:
                child = e.c
                cIdx = nodesToIdx[child]
                appendArray = np.array([pts[idx], pts[cIdx], [np.nan, np.nan]])
                edges = np.append(edges, appendArray, axis=0)

    # perform plotting.
    plt.plot(edges[:,0], edges[:,1],zorder=1)
    plt.scatter(pts[:,0], pts[:,1], s=2000.0, facecolors='white', edgecolors='red', zorder=2) #marker=plt.markers.MarkerStyle('o', fillstyles='none'))

    if show_labels:
        for n in nodesToIdx:
            idx = nodesToIdx[n]
            plt.text(pts[idx,0], pts[idx,1], str(n.getLabel()), \
                horizontalalignment='center', verticalalignment='center', fontsize=8, zorder=3)



#

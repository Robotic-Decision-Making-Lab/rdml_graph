# Copyright 2021 Ian Rankin
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
# GraphVizPlot.py
# Written Ian Rankin - April 2021
#
# An updated set of code to allow plotting using graphviz rather than
# matplotlib plots.
# This is useful for generic graphs and tree's without geometric data attached to them


import numpy as np
from rdml_graph.core import TreeNode

# directional graph
from graphviz import Digraph


# plotTree
# plot's a tree using a BFS search through the environment.
# Each level is given equal space.
# @param root - the top of the tree given as a Node
# @param max_levels - the max number of levels of the tree to plot.
# @param show_labels - shows the labels the nodes using the id of node.
def plotTree_viz(root, max_levels=-1, show_labels=False):
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



#

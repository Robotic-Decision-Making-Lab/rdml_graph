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
# HomotopyFeaturePRM.py
# Written Ian Rankin - March 2020
#
# This example shows an empty 2D world with homotopy nodes. Then an
# A* algorithm is ran on the homotopy graph
# finding the shortest point between two points on the PRM using the same h signature.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


############### Create PRM
map = {'width': 20, 'height': 20, 'hazards': np.array([[5.0, 5.0], [7.5, 3.0]])}

startN = gr.GeometricNode(0, np.array([6, 7]))
endN = gr.GeometricNode(1, np.array([8.5, 7]))

feat1 = gr.FeatureNode(2, "shaw island", pt=np.array([4.0, 8.0]), keywords={'shaw', 'island', 'isle'})
feat2 = gr.FeatureNode(3, "uf-1", pt=np.array([3.0, 3.0]), keywords={'upwelling front', 'upwelling', 'front', 'coastal front', 'coastal upwelling front'})


initialNodes = [startN, endN, feat1, feat2]

G = gr.PRM(map, 100, 6.0, connection=gr.HEdgeConn, initialNodes=initialNodes)


############### Setup and run AStar
num_features = map['hazards'].shape[0]
# Create the start homotopy node over the PRM graph.
names = frozenset(['shaw island'])
start = gr.HomotopyFeatureState(G[0], gr.HomologySignature(num_features), root=G[0], neededNames=names)


# Create the goal h signature.
goalPartialHSign = gr.HomologySignatureGoal(num_features)
goalPartialHSign.addConstraint(0, -1) # add constraints to goal hsign
goalPartialHSign.addConstraint(1, 0)

keywords = {'upwelling front'}

# A simple euclidean distance huerestic for the AStar algorithm
# I doubt this does particulary much to speed on computation if any at all.
def h_euclidean_tuple(n, data, goal):
    return np.linalg.norm(n.node.pt - goal[0].pt)

# run the AStar planning
path, cost = gr.AStar(start, g=gr.partial_homology_feature_goal, \
                        h = h_euclidean_tuple,
                        goal = (G[1], goalPartialHSign, names, keywords))

################ Output results

print('cost = ' + str(cost))

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plotHomotopyPath(path, 'red')
gr.plotFeatureNodes([feat1, feat2])
plt.scatter(map['hazards'][:,0], map['hazards'][:,1], color='blue', zorder=5)
plt.show()

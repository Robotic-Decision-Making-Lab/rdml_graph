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
map = {'size': np.array([10,10]), 'hazards': np.array([[5.0, 5.0], [7.5, 3.0]])}

startN = gr.GeometricNode(0, np.array([6, 7]))
endN = gr.GeometricNode(1, np.array([8.5, 7]))

feat1 = gr.FeatureNode(2, "shaw island", pt=np.array([4.0, 8.0]), keywords={'shaw', 'island', 'isle'})
feat2 = gr.FeatureNode(3, "uf-1", pt=np.array([3.0, 3.0]), keywords={'upwelling front', 'upwelling', 'front', 'coastal front', 'coastal upwelling front'})


initialNodes = [startN, endN, feat1, feat2]

G = gr.PRM(map, 100, 3.0, connection=gr.HomotopyEdgeConn, initialNodes=initialNodes)


############### Setup and run AStar
num_features = map['hazards'].shape[0]
# Create the start homotopy node over the PRM graph.
start = gr.HomotopyFeatureState(G[0], gr.HSignature(num_features), root=G[0])


# Create the goal h signature.
goalPartialHSign = gr.HSignatureGoal(num_features)
goalPartialHSign.addConstraint(0, -1) # add constraints to goal hsign
goalPartialHSign.addConstraint(1, 0)

names = set(['shaw island'])
keywords = {'upwelling front'}

def h_euclidean_tuple(n, data, goal):
    return np.linalg.norm(n.node.pt - goal[0].pt)

# run the AStar planning
path, cost = gr.AStar(start, g=gr.partial_homotopy_feature_goal, \
                        h = h_euclidean_tuple,
                        goal = (G[1], goalPartialHSign, names, keywords))

################ Output results

print('cost = ' + str(cost))

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plotHomotopyPath(path, 'red')
plt.scatter(map['hazards'][:,0], map['hazards'][:,1], color='blue', zorder=5)
plt.show()

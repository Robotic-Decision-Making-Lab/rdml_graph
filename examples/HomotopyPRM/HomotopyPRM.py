# HomotopyPRM.py
# Written Ian Rankin - March 2020
#
# This example shows an empty 2D world with homotopy nodes. Then an
# A* algorithm is ran on the homotopy graph
# finding the shortest point between two points on the PRM using the same h signature.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt



map = {'size': np.array([10,10]), 'features': np.array([[5.0, 5.0]])}

G = gr.PRM(map, 100, 3.0, connection=gr.HomotopyEdgeConn)


num_features = map['features'].shape[0]
# Create the start homotopy node over the PRM graph.
start = gr.HomotopyNode(G[0], gr.HSignature(num_features), root=G[0])

# Create the goal h signature.
goalHSign = gr.HSignature(num_features)
goalHSign.cross(0,1)
goal = gr.HomotopyNode(G[50], goalHSign, root=G[0])

# run the AStar planning
path, cost = gr.AStar(start, goal = goal)

print('cost = ' + str(cost))

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plotHomotopyPath(path, 'red')
plt.scatter(map['features'][:,0], map['features'][:,1], color='blue')
plt.show()

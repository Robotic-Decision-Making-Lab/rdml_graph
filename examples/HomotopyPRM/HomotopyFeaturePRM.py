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

feat1 = gr.FeatureNode(2, "Shaw Island", pt=np.array([4.0, 7.0]), keywords={'Shaw', 'Island', 'Isle'})
feat2 = gr.FeatureNode(3, "uf-1", pt=np.array([3.0, 3.0]), keywords={'Upwelling Front', 'Upwelling', 'Front', 'Coastal Front', 'Coastal upwelling front'})


initialNodes = [startN, endN, feat1, feat2]

G = gr.PRM(map, 100, 3.0, connection=gr.HomotopyEdgeConn, initialNodes=[startN, endN])


############### Setup and run AStar
num_features = map['hazards'].shape[0]
# Create the start homotopy node over the PRM graph.
start = gr.HomotopyNode(G[0], gr.HSignature(num_features), root=G[0])


# Create the goal h signature.
goalPartialHSign = gr.HSignatureGoal(num_features)
goalPartialHSign.addConstraint(0, -1) # add constraints to goal hsign
goalPartialHSign.addConstraint(1, 0)


# run the AStar planning
path, cost = gr.AStar(start, g=gr.partial_homotopy_goal_check, \
                        goal = (G[1], goalPartialHSign))

################ Output results

print('cost = ' + str(cost))

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plotHomotopyPath(path, 'red')
plt.scatter(map['hazards'][:,0], map['hazards'][:,1], color='blue', zorder=5)
plt.show()

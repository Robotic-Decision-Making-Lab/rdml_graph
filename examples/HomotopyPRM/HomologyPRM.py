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
# HomotopyPRM.py
# Written Ian Rankin - March 2020
#
# This example shows an empty 2D world with homotopy nodes. Then an
# A* algorithm is ran on the homotopy graph
# finding the shortest point between two points on the PRM using the same h signature.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import time

############### Create PRM
map = {'width': 20, 'height': 20, 'hazards': np.array([[5.0, 5.0], [7.5, 3.0]])}

startN = gr.GeometricNode(0, np.array([6, 7]))
endN = gr.GeometricNode(1, np.array([8.5, 7]))

G = gr.PRM(map, 100, 6.0, connection=gr.HEdgeConn, initialNodes=[startN, endN])


############### Setup and run AStar
num_features = map['hazards'].shape[0]
# Create the start homotopy node over the PRM graph.
start = gr.HNode(G[0], gr.HomologySignature(num_features), root=G[0])


# Create the goal h signature.
goalPartialHSign = gr.HomologySignatureGoal(num_features)
goalPartialHSign.addConstraint(0, -1) # add constraints to goal hsign
goalPartialHSign.addConstraint(1, 0)

start_t = time.time()

# run the AStar planning
path, cost, root = gr.AStar(start, g=gr.partial_h_goal_check, \
                        goal = (G[1], goalPartialHSign), output_tree=True)

end = time.time()
print('Time to execute A*')
print(str(end - start_t) +'sec')

#t = root.get_viz()
#t.view()

################ Output results

print('cost = ' + str(cost))

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plotHomotopyPath(path, 'red')
plt.scatter(map['hazards'][:,0], map['hazards'][:,1], color='blue', zorder=5)
plt.show()

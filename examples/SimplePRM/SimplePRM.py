# SimplePRM.py
# Written Ian Rankin - March 2020
#
# This example shows how a very basic 2D PRM could be implemented with a
# A* algorithm finding the shortest point between two points on the PRM.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt



map = {'size': np.array([10,10])}

G = gr.PRM(map, 100, 3.0)

path, cost = gr.AStar(G[0], goal = G[50])

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plot2DGeoPath(path, 'red')
plt.show()

# TestHomotopyGraph.py
# Written Ian Rankin February 2020
#
# A basic test script for the homotopy graph
# Designed to make sure the basic code works.

import rdml_graph as gr
import numpy as np

import pdb

features = np.array([[0,0]])
angle = np.pi / 2.0

n = gr.GeometricNode(0,np.array([-3,-3]))
n1 = gr.GeometricNode(1, np.array([-3.3,3]))
n2 = gr.GeometricNode(2, np.array([3.4,-2]))
n3 = gr.GeometricNode(3, np.array([4.4,-3]))
n4 = gr.GeometricNode(4, np.array([-2.2,-10]))
n5 = gr.GeometricNode(5, np.array([4.5,3.2]))


n.addEdge(gr.HomotopyEdge(n,n1, features=features))
n1.addEdge(gr.HomotopyEdge(n1,n, features=features))

n.addEdge(gr.HomotopyEdge(n,n2, features=features))
n2.addEdge(gr.HomotopyEdge(n2,n, features=features))

n.addEdge(gr.HomotopyEdge(n,n3, features=features))
n3.addEdge(gr.HomotopyEdge(n3,n, features=features))

n2.addEdge(gr.HomotopyEdge(n2,n4, features=features))
n4.addEdge(gr.HomotopyEdge(n4,n2, features=features))

n1.addEdge(gr.HomotopyEdge(n1,n5, features=features))
n5.addEdge(gr.HomotopyEdge(n5,n1, features=features))

n3.addEdge(gr.HomotopyEdge(n3,n5, features=features))
n5.addEdge(gr.HomotopyEdge(n5,n3, features=features))

n.addEdge(gr.HomotopyEdge(n,n5, features=features))
n5.addEdge(gr.HomotopyEdge(n5,n, features=features))

print(features.shape)

start = gr.HomotopyNode(n, gr.HSignature(1), root=n)
goalH = gr.HSignature(1)
goalH.cross(0,1)
goal = gr.HomotopyNode(n5, goalH, root=n)

print(start)
print(goal)

#pdb.set_trace()

path, cost = gr.AStar(start, goal=goal)

print('Cost = ' + str(cost))
print('Path executed')
for i in range(len(path)):
    n = path[i]
    print(n)

################## test path with partial H-sign goal
hSignGoal = gr.HSignatureGoal(1)
hSignGoal.mask[0] = 0
hSignGoal.sign[0] = 1

path, cost = gr.AStar(start, g=gr.partial_homotopy_goal_check, goal=(n5, hSignGoal))

print('HSignatureGoal used instead')
print('Cost = ' + str(cost))
print('Path executed')
for i in range(len(path)):
    n = path[i]
    print(n)

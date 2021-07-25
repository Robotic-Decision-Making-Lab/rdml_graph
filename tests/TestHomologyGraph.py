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

sign = gr.HomologySignature(features.shape[0])
n.addEdge(gr.HEdge(n,n1, sign, features=features))
n1.addEdge(gr.HEdge(n1,n, sign, features=features))

n.addEdge(gr.HEdge(n,n2, sign, features=features))
n2.addEdge(gr.HEdge(n2,n, sign, features=features))

n.addEdge(gr.HEdge(n,n3, sign, features=features))
n3.addEdge(gr.HEdge(n3,n, sign, features=features))

n2.addEdge(gr.HEdge(n2,n4, sign, features=features))
n4.addEdge(gr.HEdge(n4,n2, sign, features=features))

n1.addEdge(gr.HEdge(n1,n5, sign, features=features))
n5.addEdge(gr.HEdge(n5,n1, sign, features=features))

n3.addEdge(gr.HEdge(n3,n5, sign, features=features))
n5.addEdge(gr.HEdge(n5,n3, sign, features=features))

n.addEdge(gr.HEdge(n,n5, sign, features=features))
n5.addEdge(gr.HEdge(n5,n, sign, features=features))

print(features.shape)

start = gr.HNode(n, gr.HomologySignature(1), root=n)
goalH = gr.HomologySignature(1)
goalH.cross(0,1)
goal = gr.HNode(n5, goalH, root=n)

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
hSignGoal = gr.HomologySignatureGoal(1)
hSignGoal.mask[0] = 0
hSignGoal.sign[0] = 1

path, cost = gr.AStar(start, g=gr.partial_h_goal_check, goal=(n5, hSignGoal))

print('HomologySignatureGoal used instead')
print('Cost = ' + str(cost))
print('Path executed')
for i in range(len(path)):
    n = path[i]
    print(n)

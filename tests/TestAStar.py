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
# TestAStar.py
# Written Ian Rankin February 2020
#
# A basic set of test function to ensure that the GraphSearch algorithms are
# working properly.

import rdml_graph as gr
import pdb

n = gr.Node(0)
n1 = gr.Node(1)
n2 = gr.Node(2)
n3 = gr.Node(3)
n4 = gr.Node(4)
n5 = gr.Node(5)
n6 = gr.Node(6)


n.addEdge(gr.Edge(n,n1, 5.0))
n.addEdge(gr.Edge(n,n2, 2.7))
n.addEdge(gr.Edge(n,n3, 8.4))
n.addEdge(gr.Edge(n,n6, 34.4))
n2.addEdge(gr.Edge(n2,n4, 11.4))
n1.addEdge(gr.Edge(n1,n5,1.2))
n4.addEdge(gr.Edge(n4,n6,2.2))


correctSolution = [0,2,4,6]
path, cost, root = gr.AStar(n, goal=n6, output_tree=True)

print('Cost = ' + str(cost))
print('Path executed')
for i in range(len(path)):
    n = path[i]
    print(n)

correct = True
if len(correctSolution) == len(path):
    for i in range(len(path)):
        if correctSolution[i] != path[i].id:
            print('ID incorrect path id:'+ str(path[i].id) + ', solId:'+str(correctSolution[i]))
            correct = False
else:
    correct = False

if correct:
    print('Passed test')
else:
    print('Failed test')

t = root.get_viz(labels=True)
t.view()

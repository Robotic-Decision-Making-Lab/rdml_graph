# TestAStar.py
# Written Ian Rankin February 2020
#
# A basic set of test function to ensure that the GraphSearch algorithms are
# working properly.

import rdml_graph as gr

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
path, cost = gr.AStar(n, goal=n6)

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

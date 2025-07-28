# test_AStar.py
# Written Ian Rankin October 2022
#
# Test functions for the AStar algorithm

import pytest

import rdml_graph as gr


def test_AStar():
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

    #print('Cost = ' + str(cost))
    #print('Path executed')
    # for i in range(len(path)):
    #     n = path[i]
    #     print(n)

    #correct = True
    assert len(correctSolution) == len(path)
    for i in range(len(path)):
        assert correctSolution[i] == path[i].id, 'ID incorrect path id:'+ str(path[i].id) + ', solId:'+str(correctSolution[i])


    # check if edges being returned is correct
    path, cost = gr.AStar(n, goal=n6, output_tree=False, keepEdges=True, keepNodes=False)

    assert len(path) == 3
    assert path[0].p.id == 0
    assert path[0].c.id == 2
    assert path[1].c.id == 4
    assert path[2].c.id == 6

    # check if nodes and edges being returned is correct
    path, cost = gr.AStar(n, goal=n6, output_tree=False, keepEdges=True, keepNodes=True)

    assert len(path) == 7
    assert path[0].id == 0
    assert path[1].p.id == 0
    assert path[1].c.id == 2
    assert path[2].id == 2
    assert path[3].p.id == 2
    assert path[3].c.id == 4
    assert path[4].id == 4
    assert path[5].p.id == 4
    assert path[5].c.id == 6
    assert path[6].id == 6
# test_BFS.py
# Written Ian Rankin - October 2022 (based on code from Feb 2020)
#
# This is a pytest set of code for testing the breadth first search code.


import pytest

import rdml_graph as gr


@pytest.fixture
def root():
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

    return n

def test_BFS(root):
    paths = gr.BFS(root, budget=5.0)


    assert paths[0][1] == 0.0 # check the first path is just itself
    assert paths[1][1] == 5.0
    assert paths[2][1] == 2.7


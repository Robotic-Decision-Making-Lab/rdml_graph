# test_mcts_simple.py
# Written Ian Rankin - December 2022
#
# A very simple test of MCTS to ensure it is vaguly working and not crashing

import pytest

import rdml_graph as gr
import numpy as np


def rewardDumb(sequence, budget, data):
    return 1, 0



def test_simple_mcts_edge():

    n = gr.Node(0)
    n1 = gr.Node(1)
    n2 = gr.Node(2)
    n3 = gr.Node(3)
    n4 = gr.Node(4)
    n5 = gr.Node(5)
    n6 = gr.Node(6)


    n.addEdge(gr.Edge(n,n1, 1))
    n.addEdge(gr.Edge(n,n2, 1))
    n.addEdge(gr.Edge(n,n3, 1))
    n.addEdge(gr.Edge(n,n6, 1))
    n2.addEdge(gr.Edge(n2,n4, 1))
    n1.addEdge(gr.Edge(n1,n5,1))
    n4.addEdge(gr.Edge(n4,n6,1))


    start = n
    solution, reward, data = gr.MCTS(start, 10, rewardDumb, budget=0.5, solutionFunc=gr.highestReward, \
                            keepEdges=True, keepNodes=True)

    assert reward == 1
    assert len(solution) == 3
    assert solution[0].id == 0
    assert solution[1].p.id == 0
    assert solution[2].id != 0
    assert solution[2].id == solution[1].c.id

    start = n
    solution, reward, data = gr.MCTS(start, 10, rewardDumb, budget=0.5, solutionFunc=gr.highestReward, \
                            keepEdges=True, keepNodes=False)

    assert reward == 1
    assert len(solution) == 1
    assert solution[0].p.id == 0

# test_mcts_simple.py
# Written Ian Rankin - December 2022
#
# A very simple test of MCTS to ensure it is vaguly working and not crashing

import pytest

import rdml_graph as gr
import numpy as np

class IncrementState(gr.State):
    def __init__(self, num):
        self.num = num

    def successor(self):
        return [(IncrementState(self.num+1), 1)]

    def __str__(self):
        return 'IncState(' + str(self.num) + ')'

def rewardDumb(sequence, budget, data):
    return 1, 0



def test_simple_mcts():
    start = IncrementState(0)
    solution, reward, data = gr.MCTS(start, 500, rewardDumb, budget=0.5, solutionFunc=gr.highestReward)

    assert reward == 1
    assert solution[0].num == 0
    assert solution[1].num == 1


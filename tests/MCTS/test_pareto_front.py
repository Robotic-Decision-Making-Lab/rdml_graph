# test_pareto_front.py
# Written Ian Rankin - December 2022
#
# A set of tests to test the pareto front code to ensure it is working properly.

import pytest

import rdml_graph as gr
import numpy as np


def test_pareto_class():
    front = gr.ParetoFront(3, alloc_size=10)



    rewards = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    for i in range(rewards.shape[0]):
        front.check_and_add(rewards[i], i)

    ans = [2,3,5]
    pr_vals, pr_idxs = front.get()
    for i in range(len(ans)):
        assert pr_idxs[i] == ans[i]



def test_pareto_function():
    front = gr.ParetoFront(3, alloc_size=10)



    rewards = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    pr_idxs = gr.get_pareto(rewards)

    ans = [2,3,5]
    for i in range(len(ans)):
        assert pr_idxs[i] == ans[i]



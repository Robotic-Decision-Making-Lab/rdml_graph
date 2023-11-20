# test_active_learners.py
# Written Ian Rankin - November 2023
#
# A set of tests to ensure that active learning
# still works properly in certian sub cases

import pytest

import numpy as np
import rdml_graph as gr


def test_det_learner():
    learner = gr.DetLearner(alpha=1.0)
    assert isinstance(learner, gr.ActiveLearner)

def test_UCB_learner():
    learner = gr.UCBLearner(alpha=1.0)
    assert isinstance(learner, gr.ActiveLearner)

def test_random_learner():
    learner = gr.RandomLearner()
    assert isinstance(learner, gr.ActiveLearner)

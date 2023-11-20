# test_active_gp.py
# Written Ian Rankin - November 2023
#
# A set of tests to ensure that active learning
# still works properly in certian sub cases

import pytest

import numpy as np
import rdml_graph as gr


def f_sin(x, data=None):
    return 2 * np.sum(np.cos(np.pi * (x-2)) * np.exp(-(0.9*x)), axis=1)

def f_lin(x, data=None):
    #return x[:,0]*x[:,1]
    return x[:,0]+x[:,1]

def f_sq(x, data=None):
    return x[:,0]*x[:,0] + 1.2*x[:,1]

def test_adding_prior_no_pareto_pairs():
    bounds = [(0,7), (0,7)]
    gp = gr.PreferenceGP(gr.RBF_kern(1.0, 1.0), pareto_pairs=False, \
                        use_hyper_optimization=False, \
                        active_learner = gr.DetLearner(1.0))
    gp.add_prior(bounds=np.array(bounds), num_pts=20)

def active_learner_no_prior(utility_f, learner):
    num_side = 25
    bounds = [(0,7), (0,7)]

    num_train_pts = 40
    num_alts = 3


    gp = gr.PreferenceGP(gr.RBF_kern(1.0, 1.0), pareto_pairs=False, \
                        use_hyper_optimization=False, \
                        active_learner = learner)

    for i in range(10):
        train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
        train_Y = utility_f(train_X)#f_lin(train_X)

        #pdb.set_trace()
        selected_idx, UCB, best_value = gp.select(train_X, num_alts)
        #selected_idx = gp.active_learner.select_previous(train_X, num_alts=num_alts)

        best_idx = np.argmax(train_Y[selected_idx])

        #import pdb
        #pdb.set_trace()
        pairs = gr.ranked_pairs_from_fake(train_X[selected_idx], utility_f)

        gp.add(train_X[selected_idx], pairs)



def test_active_no_prior_sin_det_learner():
    active_learner_no_prior(f_sin, gr.DetLearner(1.0))

def test_active_no_prior_lin_det_learner():
    active_learner_no_prior(f_lin, gr.DetLearner(1.0))

def test_active_no_prior_sq_det_learner():
    active_learner_no_prior(f_sq, gr.DetLearner(1.0))

def test_active_no_prior_sin_random_learner():
    active_learner_no_prior(f_sin, gr.RandomLearner())

def test_active_no_prior_lin_random_learner():
    active_learner_no_prior(f_lin, gr.RandomLearner())

def test_active_no_prior_sq_random_learner():
    active_learner_no_prior(f_sq, gr.RandomLearner())

def test_active_no_prior_sin_mutual_learner():
    active_learner_no_prior(f_sin, gr.MutualInformationLearner())

def test_active_no_prior_lin_mutual_learner():
    active_learner_no_prior(f_lin, gr.MutualInformationLearner())

def test_active_no_prior_sq_mutual_learner():
    active_learner_no_prior(f_sq, gr.MutualInformationLearner())



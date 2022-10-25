# test_shap_calc.py
# Written Ian Rankin - October 2022
#
# 

import pytest

import rdml_graph as gr
import matplotlib.pyplot as plt
import numpy as np
import random


# This function generates a decision tree with a leaf node off of the root
# This is used as I suspect one of my bugs is coming from a single node
@pytest.fixture
def root_single():
    X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1]])
    types = ['float'] * len(X_in[0])
    Y = X_in[:,0] + 0.7*X_in[:,1] + X_in[:,2]**2 + X_in[:,3]
    #X_in = X_in.tolist()
    #Y = Y.tolist()

    X = list(zip(X_in, Y))

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100)

    return root

# This function generates a decision tree with a leaf node off of the root
# This is used as I suspect one of my bugs is coming from a single node
@pytest.fixture
def forest_single():
    X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1]])
    types = ['float'] * len(X_in[0])
    Y = X_in[:,0] + 0.7*X_in[:,1] + X_in[:,2]**2 + X_in[:,3]
    #X_in = X_in.tolist()
    #Y = Y.tolist()

    X = list(zip(X_in, Y))

    root,_ = gr.learn_random_forest(X, \
                    num_trees=20, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100, \
                    num_threads=4)

    return root

# This function generates a decision tree with a leaf node off of the root
# This is used as I suspect one of my bugs is coming from a single node
@pytest.fixture
def forest_multi():
    X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1], [0.5, 0.2, -1, -2], [3,1,2,1]])
    types = ['float'] * len(X_in[0])
    Y = X_in[:,0] + 0.7*X_in[:,1] + X_in[:,2]**2 + X_in[:,3]
    #X_in = X_in.tolist()
    #Y = Y.tolist()

    X = list(zip(X_in, Y))

    root,_ = gr.learn_random_forest(X, \
                    num_trees=20, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100, \
                    num_threads=4)

    return root

# Generates a simple regression decision tree to test shapely value calculation using.
@pytest.fixture
def root_multi():
    X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1], [0.5, 0.2, -1, -2], [3,1,2,1]])
    types = ['float'] * len(X_in[0])
    Y = X_in[:,0] + 0.7*X_in[:,1] + X_in[:,2]**2 + X_in[:,3]
    #X_in = X_in.tolist()
    #Y = Y.tolist()

    X = list(zip(X_in, Y))

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100)

    return root

def test_single_shap_no_int_node_single(root_single):
    x = np.array([0,0,0,0])
    shap = gr.TreeSHAP(x, root_single)
    print(shap)

    pre = 22
    assert sum(shap) < 0 + pre
    assert sum(shap) > 0 - pre

    x = np.array([0,1,2,3])
    shap = gr.TreeSHAP(x, root_single)
    print(shap)

    pre = 30
    assert sum(shap) < 7.7 + pre
    assert sum(shap) > 7.7 - pre

    x = np.array([1,2,3,4])
    shap = gr.TreeSHAP(x, root_single)
    print(shap)

    pre = 30
    assert sum(shap) < 15.4 + pre
    assert sum(shap) > 15.4 - pre

def test_single_shap_no_int(root_multi):
    x = np.array([0,0,0,0])
    shap = gr.TreeSHAP(x, root_multi)
    print(shap)

    pre = 20
    assert sum(shap) < 0 + pre
    assert sum(shap) > 0 - pre

    x = np.array([1,2,3,4])
    shap = gr.TreeSHAP(x, root_multi)
    print(shap)

    pre = 20
    assert sum(shap) < 15.4 + pre
    assert sum(shap) > 15.4 - pre


def test_single_shap_int_node_single(root_single):
    x = np.array([0,0,0,0])
    shap = gr.TreeSHAP_INT(x, root_single)
    print(shap)

    pre = 0.5
    assert sum(shap) < 0 + pre
    assert sum(shap) > 0 - pre

    x = np.array([0,1,2,3])
    shap = gr.TreeSHAP_INT(x, root_single)
    print(shap)

    pre = 2
    assert sum(shap) < 7.7 + pre
    assert sum(shap) > 7.7 - pre

    x = np.array([1,2,3,4])
    shap = gr.TreeSHAP_INT(x, root_single)
    print(shap)

    pre = 4
    assert sum(shap) < 15.4 + pre
    assert sum(shap) > 15.4 - pre

def test_single_shap_int(root_multi):
    x = np.array([0,0,0,0])
    shap = gr.TreeSHAP_INT(x, root_multi)
    print(shap)

    pre = 0.5
    assert sum(shap) < 0 + pre
    assert sum(shap) > 0 - pre

    x = np.array([1,2,3,4])
    shap = gr.TreeSHAP_INT(x, root_multi)
    print(shap)

    pre = 4
    assert sum(shap) < 15.4 + pre
    assert sum(shap) > 15.4 - pre



def test_shap_all(root_multi):
    Y = np.array([7.7, 15.4, 0.0, 3.7, 4.7, -0.3599, 8.7])

    shap = gr.SHAP_all(root_multi)
    sums = np.sum(shap, axis=1)

    pre = 5
    for i, y in enumerate(Y):
        assert y < sums[i] + pre
        assert y > sums[i] - pre

def test_shap_all_single(root_single):
    Y = np.array([7.7, 15.4, 0.0, 3.7, 4.7])

    shap = gr.SHAP_all(root_single)
    sums = np.sum(shap, axis=1)

    pre = 5
    for i, y in enumerate(Y):
        assert y < sums[i] + pre
        assert y > sums[i] - pre

def test_shap_all_forest(forest_multi):
    Y = np.array([7.7, 15.4, 0.0, 3.7, 4.7, -0.3599, 8.7])

    shap = gr.SHAP_all(forest_multi)
    sums = np.sum(shap, axis=1)

    pre = 10
    for i, y in enumerate(Y):
        assert y < sums[i] + pre
        assert y > sums[i] - pre

def test_shap_all_forest_single(forest_single):
    Y = np.array([7.7, 15.4, 0.0, 3.7, 4.7])

    shap = gr.SHAP_all(forest_single)
    sums = np.sum(shap, axis=1)

    pre = 10
    for i, y in enumerate(Y):
        assert y < sums[i] + pre
        assert y > sums[i] - pre



if __name__ == '__main__':
    X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1]])
    #X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1], [0.5, 0.2, -1, -2], [3,1,2,1]])
    types = ['float'] * len(X_in[0])
    Y = X_in[:,0] + 0.7*X_in[:,1] + X_in[:,2]**2 + X_in[:,3]
    #X_in = X_in.tolist()
    #Y = Y.tolist()

    X = list(zip(X_in, Y))

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100)
    
    t = root.get_viz()
    t.view()

    x = np.array([0,0,0,0])
    shap = gr.TreeSHAP(x, root)
    shap = gr.TreeSHAP_INT(x, root)
    shap = gr.SHAP_all(root)
    sums = np.sum(shap, axis=1)
    print(shap)
    print(sums)

    
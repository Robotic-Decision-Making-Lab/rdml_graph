# test_decision_tree.py
# Written Ian Rankin - October 2022
#
# A set of pytest tests for the decision tree classes.

import pytest

import rdml_graph as gr
import sklearn.datasets as dt
import numpy as np
import random

@pytest.fixture
def iris():
    iris = dt.load_iris()
    X = iris.data
    y = iris.target
    return (X,y)



def test_balanced_tree(iris):
    X,y = iris[0], iris[1]
    root, _ = gr.create_balanced_decision_tree(X)

    for x in X:
        ans = root.traverse(x)

        assert (ans == x).all()



def test_iris_classification(iris):
    X_in, y = iris[0], iris[1]
    types = ['float'] * len(X_in[0])

    X = list(zip(X_in, y))

    # root,_ = gr.learn_random_forest(X, \
    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.classification_importance, \
                    plurality_func=gr.class_plurality,\
                    max_depth=100)

    for x,y_i in X:
        ans = root.traverse(x)

        assert (ans == y_i).all()


    
def test_regression_test():
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

    for x,y_i in X:
        ans = root.traverse(x)

        assert (ans == y_i).all()


def test_regression_test_with_labels():
    X_in = np.array([[0,1,2,3], [1,2,3,4], [0,0,0,0], [1,1,1,1], [2,1,1,1]])
    types = ['float'] * len(X_in[0])
    Y = X_in[:,0] + 0.7*X_in[:,1] + X_in[:,2]**2 + X_in[:,3]
    #X_in = X_in.tolist()
    #Y = Y.tolist()

    Y = [(y, ('hello',)) for y in Y]

    X = list(zip(X_in, Y))

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    with_labels=True,\
                    max_depth=100)

    for x,y_i in X:
        ans = root.traverse(x)
        assert ans[0] == y_i[0]






def f(x_in):
    y_0 = np.sin(x_in[0]-.3572)*4-0.2
    y_1 = (np.cos(x_in[1]*1.43)-.3572)*3
    #y_2 = x_in[2]

    return y_0 + y_1# + y_2

def test_multivariate_tree():
    num_samps = 100
    num_dim = 2

    xs = [[random.uniform(0,10) for j in range(num_dim)]  for i in range(num_samps)]
    X = [[x, f(x)+random.uniform(-0.5, 0.5)] for x in xs]

    #print(X)
    #X = np.array(X)
    #print(xs)
    #print(X)

    types = ['float'] * num_dim

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100)

    for x,y_i in X:
        ans = root.traverse(x)

        assert (ans == y_i).all()


if __name__ == '__main__':
    test_multivariate_tree()



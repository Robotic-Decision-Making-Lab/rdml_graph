# test_decision_tree.py
# Written Ian Rankin - October 2022
#
# A set of pytest tests for the decision tree classes.

import pytest

import rdml_graph as gr
import sklearn.datasets as dt

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



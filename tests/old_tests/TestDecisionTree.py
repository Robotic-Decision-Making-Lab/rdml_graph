# Copyright 2021 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# TestDecisionTree.py
# Written Ian Rankin April 2021
#
# A set of tests for decision tree to make sure it is working.

import rdml_graph as gr

import sklearn.datasets as dt
import numpy as np
import matplotlib.pyplot as plt

import random

if __name__ == '__main__':
    X = [([1, 'a'], 0), ([5, 'a'], 0), ([27, 'a'], 0), ([12, 'b'], 1), ([8, 'b'], 2)]
    types = ['float', 'category']

    root, _ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.classification_importance)

    #x = ['b', 'c']
    x = [4.5, 'b']
    prediction = root.traverse(x)
    print(prediction)

    t = root.get_viz(labels=True)
    t.view()

    ########################### Iris dataset

    iris = dt.load_iris()

    X = list(zip(iris.data, iris.target))
    types = ['float', 'float', 'float', 'float']

    root, _ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.classification_importance)

    x = [3,2,5,6]
    prediction = root.traverse(x)
    print(prediction)

    t = root.get_viz(labels=True)
    t.view()

    ######################### Regression  decision tree (sin function)

    # test regression
    xs = [random.uniform(0,10) for i in range(2000)]
    X = [([x, 'a'], np.sin(x-.3572)*4-0.2) for x in xs]
    types = ['float', 'category']

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    #importance_func=gr.regression_importance, \
                    importance_func=gr.least_squares_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=4)

    reg_root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    #importance_func=gr.least_squares_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=4)

    from graphviz import Digraph

    x = [3.4, 'a']
    y = root.traverse(x)
    print(y)

    t = Digraph('T')
    t2 = root.get_viz(labels=True, t=t)
    t2.view()

    t = Digraph('T-regres')
    t2 = reg_root.get_viz(labels=True, t=t)
    t2.view()


    xs = np.arange(0,10,0.01)
    y = [root.traverse([i, 'a']) for i in xs]

    #print(y)
    plt.plot(xs, y)
    #plt.plot([-2, 12], [-2*2.1 - 0.4, 12*2.1-0.4])
    plt.show()

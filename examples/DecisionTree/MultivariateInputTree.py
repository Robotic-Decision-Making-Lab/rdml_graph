# MultivariateInputTree.py
# Written Ian Rankin - June 2021
#
# An example of using the decision tree for regression
# with multiple inputs.


import rdml_graph as gr

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

def f(x_in):
    y_0 = np.sin(x_in[0]-.3572)*4-0.2
    y_1 = (np.cos(x_in[1]*1.43)-.3572)*3
    #y_2 = x_in[2]

    return y_0 + y_1# + y_2


def main():
    num_samps = 100
    num_dim = 2

    xs = [[random.uniform(0,10) for j in range(num_dim)]  for i in range(num_samps)]
    X = [(x, f(x)+random.uniform(-0.5, 0.5)) for x in xs]

    X = np.array(X)
    #print(xs)
    #print(X)

    types = ['float'] * num_dim

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    with_labels=False, \
                    max_depth=100)

    t = root.get_viz(labels=True)
    t.view()

    xs = np.arange(0,10,0.1)
    xs = np.transpose([np.tile(xs, len(xs)), np.repeat(xs, len(xs))])
    #print(xs)
    xs = list(xs)
    y_pred = [root.traverse(x) for x in xs]
    y_actual = [f(x) for x in xs]

    #print(y)
    #plt.plot(xs, y)
    #plt.plot([-2, 12], [-2*2.1 - 0.4, 12*2.1-0.4])
    #plt.show()
    ax = plt.axes(projection='3d')

    x_0 = np.array([x[0] for x in xs])
    x_1 = np.array([x[1] for x in xs])

    y_pred = np.array(y_pred)
    #ax.contour3D(x_0, x_1, y_pred, 50, cmap='viridis', linewidth=0.5)
    #ax.scatter(x_0, x_1, y_pred, c=y_pred, cmap='viridis', linewidth=0.5)
    ax.scatter(x_0, x_1, y_actual, c=y_actual, cmap='magma', linewidth=0.5)
    #ax.scatter([x[0][0] for x in X], [x[0][1] for x in X], [x[1] for x in X], c=[x[1] for x in X], cmap='plasma', linewidth=1.0)

    plt.show()


if __name__ == '__main__':
    main()

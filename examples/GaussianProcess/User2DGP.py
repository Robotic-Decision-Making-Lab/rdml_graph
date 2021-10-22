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

# SimpleGP.py
# Written Ian Rankin - September 2021
#
# An example usage of a simple Gaussian process

import numpy as np
import matplotlib.pyplot as plt

import rdml_graph as gr
import pdb

def f_sin(x, data=None):
    x = 6-x
    return 2 * np.cos(np.pi * (x[:,0]-2)) * np.exp(-(0.9*x[:,0])) + 1.5 * np.cos(np.pi * (x[:,1]-2)) * np.exp(-(1.2*x[:,1]))

def f_lin(x, data=None):
    return x[:,0]*x[:,1]





if __name__ == '__main__':
    num_side = 25
    bounds = [(0,7), (0,7)]

    num_train_pts = 50

    train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
    train_Y = f_sin(train_X)

    gp = gr.GP(gr.RBF_kern(1,0.7))


    for i in range(1):
        train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
        train_Y = f_sin(train_X)

        selected_idx, UCB, best_value = gp.ucb_selection(train_X, 5)

        pdb.set_trace()
        gp.add(train_X[selected_idx], train_Y[selected_idx])



    x = np.linspace(bounds[0][0], bounds[0][1], num_side)
    y = np.linspace(bounds[1][0], bounds[1][1], num_side)

    X, Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).transpose()
    z = f_sin(points)
    Z = np.reshape(z, (num_side, num_side))

    z_predicted, z_sigma = gp.predict(points)
    Z_pred = np.reshape(z_predicted, (num_side, num_side))


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z_pred, 50, cmap='binary')
    ax.plot_wireframe(X, Y, Z, color= 'black')
    ax.plot_surface(X, Y, Z_pred, rstride=1, cstride=1, cmap='magma', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

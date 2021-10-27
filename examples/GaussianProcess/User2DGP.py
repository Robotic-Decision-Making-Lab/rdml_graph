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

# User2DGP.py
# Written Ian Rankin - September 2021
#
# An example usage of a preference Gaussian process with 2D inputs.

import numpy as np
import matplotlib.pyplot as plt

import rdml_graph as gr
import tqdm
import pdb

def f_sin(x, data=None):
    x = 6-x
    return 2 * np.cos(np.pi * (x[:,0]-2)) * np.exp(-(0.9*x[:,0])) + 1.5 * np.cos(np.pi * (x[:,1]-2)) * np.exp(-(1.2*x[:,1]))

def f_lin(x, data=None):
    #return x[:,0]*x[:,1]
    return x[:,0]+x[:,1]

def f_sq(x, data=None):
    return x[:,0]*x[:,0] + 1.2*x[:,1]



if __name__ == '__main__':
    num_side = 25
    bounds = [(0,7), (0,7)]

    num_train_pts = 40
    num_alts = 4

    utility_f = f_sq

    #train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
    #train_Y = f_sin(train_X)

    #gp = gr.PreferenceGP(gr.RBF_kern(0.2,0.5)*gr.linear_kern(0.2, 0.1, 0))
    #gp = gr.PreferenceGP(gr.linear_kern(0.3, 0.1, 0.0))
    gp = gr.PreferenceGP(gr.RBF_kern(0.5, 1.0), pareto_pairs=True)
    gp.add_prior(bounds=np.array(bounds))


    # train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
    # train_Y = utility_f(train_X)#f_lin(train_X)
    #
    # pairs = []
    # for i in range(len(train_X)):
    #     pairs += gr.generate_fake_pairs(train_X, utility_f, i)
    # gp.add(train_X, pairs)

    for i in tqdm.tqdm(range(10)):
        train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
        train_Y = utility_f(train_X)#f_lin(train_X)

        selected_idx, UCB, best_value = gp.ucb_selection(train_X, num_alts)

        best_idx = np.argmax(train_Y[selected_idx])

        #pairs = gr.gen_pairs_from_idx(best_idx, np.arange(num_alts, dtype=np.int))
        pairs = gr.ranked_pairs_from_fake(train_X[selected_idx], utility_f)

        print(pairs)
        print(train_Y[selected_idx])
        print(train_X[selected_idx])
        #best_idx = np.argmax(train_Y)
        #pairs = gr.gen_pairs_from_idx(best_idx, np.arange(num_alts, dtype=np.int))
        #print(selected_idx)
        #print(UCB)
        #print(train_X[selected_idx])
        #pdb.set_trace()
        #print(train_Y[selected_idx])

        #gp.add(train_X[selected_idx], pairs)
        gp.add(train_X[selected_idx], pairs)

    gp.optimize(optimize_hyperparameter=False)

    x = np.linspace(bounds[0][0], bounds[0][1], num_side)
    y = np.linspace(bounds[1][0], bounds[1][1], num_side)

    X, Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).transpose()
    z = utility_f(points)
    z_norm = np.linalg.norm(z, ord=np.inf)
    z = z / z_norm
    Z = np.reshape(z, (num_side, num_side))

    z_predicted, z_sigma = gp.predict(points)
    ucb_pred = z_predicted + np.sqrt(z_sigma)*1
    Z_pred = np.reshape(z_predicted, (num_side, num_side))
    UCB_pred = np.reshape(ucb_pred, (num_side, num_side))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z_pred, 50, cmap='binary')
    ax.plot_wireframe(X, Y, Z, color= 'black')
    ax.plot_wireframe(X, Y, UCB_pred, color= 'red')
    ax.plot_surface(X, Y, Z_pred, rstride=1, cstride=1, cmap='magma', edgecolor='none')
    ax.scatter(gp.X_train[:,0], gp.X_train[:,1], gp.F)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #print(gp.X_train)
    #print(gp.F)

    plt.show()

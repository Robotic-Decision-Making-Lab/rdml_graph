# test_GP.py
# Written Ian Rankin October 2022
#
# My first pytest for this repo.
# So kind of a test of pytest to make sure it does what I want it to do.

import pytest

import numpy as np
import rdml_graph as gr

import tqdm


# Test the basic GP is working properly with a couple basic tests.
def test_simple_GP_prediction():
    X_train = np.array([0,1,2,3,6,7])
    X = np.arange(-3, 12, 0.1)
    y_train = np.array([1, 0.5,0, -1, 1, 2])
    training_sigma=np.array([1, 0.5, 0.1, 0.1, 0.2, 0])

    gp = gr.GP(gr.RBF_kern(1,1)+gr.periodic_kern(1,1,10)+gr.linear_kern(3,1,0.3))
    gp.add(X_train, y_train, training_sigma=training_sigma)

    mu, sigma = gp.predict(X)

    pre = 0.5
    assert mu[30] < y_train[0]+pre
    assert mu[30] > y_train[0]-pre

    assert mu[40] < y_train[1]+pre
    assert mu[40] > y_train[1]-pre



def f_sin(x, data=None):
    x = 6-x
    return 2 * np.cos(np.pi * (x[:,0]-2)) * np.exp(-(0.9*x[:,0])) + 1.5 * np.cos(np.pi * (x[:,1]-2)) * np.exp(-(1.2*x[:,1]))

def f_lin(x, data=None):
    return x[:,0]*x[:,1]

def GP_active_learning(func):
    num_side = 25
    bounds = [(0,7), (0,7)]

    num_train_pts = 50

    train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
    train_Y = func(train_X)

    gp = gr.GP(gr.RBF_kern(1,0.8))


    for i in tqdm.tqdm(range(20)):
        train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
        train_Y = func(train_X)

        selected_idx, UCB, best_value = gp.ucb_selection(train_X, 5)

        gp.add(train_X[selected_idx], train_Y[selected_idx])


    x = np.linspace(bounds[0][0], bounds[0][1], num_side)
    y = np.linspace(bounds[1][0], bounds[1][1], num_side)

    X, Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).transpose()
    z = func(points)
    Z = np.reshape(z, (num_side, num_side))

    z_predicted, z_sigma = gp.predict(points)
    Z_pred = np.reshape(z_predicted, (num_side, num_side))

    return Z_pred

def test_GP_active_learning_linear():
    Z_lin = GP_active_learning(f_lin)

    pre = 5
    assert Z_lin[-1,-1] > 50-pre
    assert Z_lin[-1,-1] < 50+pre

def test_GP_active_learning_sin():
    Z_sin = GP_active_learning(f_sin)

    pre = 5
    assert Z_sin[-1,-1] > -10-pre
    assert Z_sin[-1,-1] < -10+pre

if __name__ == '__main__':
    test_GP_active_learning_sin()

# test_user_GP.py
# Written Ian Rankin - October 2022
#
# A set of tests for the user preferences.

import pytest

import numpy as np
import rdml_graph as gr
import tqdm


def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def f_lin(x, data=None):
    #return x[:,0]*x[:,1]
    return x[:,0]+x[:,1]

def f_sq(x, data=None):
    return x[:,0]*x[:,0] + 1.2*x[:,1]


def test_user_gp():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = gr.generate_fake_pairs(X_train, f_sin, 0) + \
            gr.generate_fake_pairs(X_train, f_sin, 1) + \
            gr.generate_fake_pairs(X_train, f_sin, 2) + \
            gr.generate_fake_pairs(X_train, f_sin, 3) + \
            gr.generate_fake_pairs(X_train, f_sin, 4)


    gp = gr.PreferenceGP(gr.RBF_kern(0.5, 0.7))
    #gp = gr.PreferenceGP(gr.periodic_kern(1.2,0.3,5))
    #gp = gr.PreferenceGP(gr.linear_kern(0.2, 0.2, 0.2))
    #gp = gr.PreferenceGP(gr.RBF_kern(0.2,1)+gr.periodic_kern(1,0.2,0)+gr.linear_kern(0.2,0.1,0.3))
    #gp = gr.PreferenceGP(gr.RBF_kern(0.1,1)*gr.linear_kern(0.3,0.2,0.3))

    gp.add(X_train, pairs)

    gp.optimize(optimize_hyperparameter=False)
    #print('gp.calc_ll()')
    #print(gp.calc_ll())


    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    y, sigma = gp.predict(X_train)

    for i in range(len(X_train)):
        if i != 0:
            assert y[0] > y[i]
        if i!= 1:
            assert y[1] < y[i]



def user_gp_active_func(utility_f):
    num_side = 25
    bounds = [(0,7), (0,7)]

    num_train_pts = 40
    num_alts = 4


    #gp = gr.PreferenceGP(gr.RBF_kern(0.2,0.5)*gr.linear_kern(0.2, 0.1, 0))
    #gp = gr.PreferenceGP(gr.linear_kern(0.3, 0.1, 0.0))
    gp = gr.PreferenceGP(gr.RBF_kern(1.0, 1.0), pareto_pairs=True, \
                        use_hyper_optimization=False, \
                        active_learner = gr.DetLearner(1.0))
    gp.add_prior(bounds=np.array(bounds), num_pts=20)



    for i in tqdm.tqdm(range(10)):
        train_X = np.random.random((num_train_pts,2)) * np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]]) + np.array([bounds[0][0], bounds[1][0]])
        train_Y = utility_f(train_X)#f_lin(train_X)

        #pdb.set_trace()
        selected_idx, UCB, best_value = gp.select(train_X, num_alts)
        #selected_idx = gp.active_learner.select_previous(train_X, num_alts=num_alts)

        best_idx = np.argmax(train_Y[selected_idx])

        pairs = gr.ranked_pairs_from_fake(train_X[selected_idx], utility_f)

        print(pairs)
        print(train_Y[selected_idx])
        print(train_X[selected_idx])


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

    
    assert UCB_pred[-1,-1] > UCB_pred[0,0]
    assert UCB_pred[-2,-2] > UCB_pred[0,0]
    assert UCB_pred[-3,-3] > UCB_pred[0,0]



def test_user_gp_active_sq():
    user_gp_active_func(f_sq)

def test_user_gp_active_lin():
    user_gp_active_func(f_lin)




def test_abs_gp_user():
    X_train = np.array([0.4, 0.7, 0.9, 1.1, 1.2, 1.35, 1.4])
    abs_values = np.array([0.999, 0.6, 0.3, 0.2, 0.22, 0.4, 0.5])
    #abs_values = np.array([0.4, 0.2, 0.2, 0.2, 0.1, 0.11, 0.3])


    gp = gr.PreferenceGP(gr.RBF_kern(0.3, 0.25), normalize_gp=True, \
            normalize_positive=True, \
            pareto_pairs=True, \
            other_probits={'abs': gr.AbsBoundProbit(1.0,10.0)})
    
    X_train = X_train[:, np.newaxis]
    gp.add(X_train[0:3], abs_values[0:3], type='abs')
    gp.add(np.array([[0.6]]), [])
    
    gp.add(X_train[3:], abs_values[3:], type='abs')

    step = 0.02
    X = np.arange(0.0, 1.5, step)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    pre = 0.3
    assert mu[int(0.4/step)] < 1 + pre
    assert mu[int(0.4/step)] > 1 - pre
    assert mu[int(0.95/step)] < 0 + pre
    assert mu[int(0.95/step)] > 0 - pre
    assert mu[int(1.4/step)] < 0.5 + pre
    assert mu[int(1.4/step)] > 0.5 - pre



def test_user_gp_ordinal():
    X_train = np.array([0,1,2,3,4.2,6,7])
    ratings = np.array([5,5,2,1,2  ,3,3])


    gp = gr.PreferenceGP(gr.RBF_kern(0.5, 0.7), \
            other_probits={'ordinal': gr.OrdinalProbit(2.0,1.0, n_ordinals=5)})
    #gp = gr.PreferenceGP(gr.periodic_kern(1.2,0.3,5))
    #gp = gr.PreferenceGP(gr.linear_kern(0.2, 0.2, 0.2))
    #gp = gr.PreferenceGP(gr.RBF_kern(0.2,1)+gr.periodic_kern(1,0.2,0)+gr.linear_kern(0.2,0.1,0.3))
    #gp = gr.PreferenceGP(gr.RBF_kern(0.1,1)*gr.linear_kern(0.3,0.2,0.3))

    gp.add(X_train, ratings, type='ordinal')

    #gp.optimize(optimize_hyperparameter=True)
    #print('gp.calc_ll()')
    #print(gp.calc_ll())

    step = 0.1
    X = np.arange(0, 8, step)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    assert mu[int(0.5/step)] > mu[int(3/step)]
    assert mu[int(6.5/step)] > mu[int(3/step)]
    assert mu[int(0.5/step)] > mu[int(6.5/step)]


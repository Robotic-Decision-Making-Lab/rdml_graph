# test_GP_kernel.py
# Written Ian Rankin - October 2022 (Based on tests written oct. 2021)
#
# 

import pytest

import rdml_graph as gr
import numpy as np




def test_vectorization_rbf():
    k = gr.RBF_kern(1,0.8)

    X = np.array([1,3,4,5,6,7])
    Y = np.array([-1,-0.5,0,1,2,3])


    cov = np.empty((len(X), len(Y)))

    for i,x1 in enumerate(X):
        for j,x2 in enumerate(Y):
            cov_ij = k(x1, x2)
            cov[i,j] = cov_ij


    cov_vec = k.cov(X,Y)

    for i in range(cov_vec.shape[0]):
        for j in range(cov_vec.shape[1]):
            assert cov[i,j] == cov_vec[i,j]

def test_vectorization_linear():
    k = gr.linear_kern(1,0.8, 0.5)

    X = np.array([1,3,4,5,6,7])
    Y = np.array([-1,-0.5,0,1,2,3])


    cov = np.empty((len(X), len(Y)))

    for i,x1 in enumerate(X):
        for j,x2 in enumerate(Y):
            cov_ij = k(x1, x2)
            cov[i,j] = cov_ij


    cov_vec = k.cov(X,Y)

    for i in range(cov_vec.shape[0]):
        for j in range(cov_vec.shape[1]):
            assert cov[i,j] == cov_vec[i,j]

def test_vectorization_periodic():
    k = gr.periodic_kern(1,0.8, 10)

    X = np.array([1,3,4,5,6,7])
    Y = np.array([-1,-0.5,0,1,2,3])


    cov = np.empty((len(X), len(Y)))

    for i,x1 in enumerate(X):
        for j,x2 in enumerate(Y):
            cov_ij = k(x1, x2)
            cov[i,j] = cov_ij


    cov_vec = k.cov(X,Y)

    for i in range(cov_vec.shape[0]):
        for j in range(cov_vec.shape[1]):
            assert cov[i,j] == cov_vec[i,j]


def test_vectorization_dual():
    k = gr.RBF_kern(1,0.8) + (gr.periodic_kern(1,0.8, 10) * gr.linear_kern(1,0.8, 0.5))

    X = np.array([1,3,4,5,6,7])
    Y = np.array([-1,-0.5,0,1,2,3])


    cov = np.empty((len(X), len(Y)))

    for i,x1 in enumerate(X):
        for j,x2 in enumerate(Y):
            cov_ij = k(x1, x2)
            cov[i,j] = cov_ij


    cov_vec = k.cov(X,Y)

    for i in range(cov_vec.shape[0]):
        for j in range(cov_vec.shape[1]):
            assert cov[i,j] == cov_vec[i,j]




def test_rbf():
    rbf = gr.RBF_kern(1, 1)
    assert rbf(1,2) > 0.5 # probably right
    assert rbf(1,2) < 0.7 # probably right

def test_periodic_kern():
    perd = gr.periodic_kern(1,1,3)
    assert perd(1,2) > 0.15 # probably right
    assert perd(1,2) < 0.3 # probably right

def test_linear_kern():
    lin = gr.linear_kern(1,1,1)
    assert lin(1,2) == 1

def test_combined_kern():
    rbf = gr.RBF_kern(1, 1)
    kern2 = gr.periodic_kern(1,1, 3)
    kern3 = gr.linear_kern(1,1,1)

    combined = rbf + (kern2 * kern3)

    assert type(combined) is gr.dual_kern
    assert combined(1,2) > 0.8 # probably right
    assert combined(1,2) < 0.9 # probably right

    param_l = [1,1,1,1,3,1,1,1]
    for i in range(len(param_l)):
        assert combined.get_param()[i] == param_l[i]

    combined.gradient(1,2) # just check this is running not gonna check values

    param_l = [2,3,5,4,7, 1,1,2]
    combined.set_param(param_l)
    for i in range(len(param_l)):
        assert combined.get_param()[i] == param_l[i]


def test_rbf_kern_cov():
    rbf_test = gr.RBF_kern(1, 1)
    X = np.array([0,1,2,3,4.2,6,7])


    c = rbf_test.cov(X,X)

    for i in range(len(X)):
        assert c[i,i] == 1
    
    assert c[-1,0] < 0.0001
    assert c[0,-1] < 0.0001
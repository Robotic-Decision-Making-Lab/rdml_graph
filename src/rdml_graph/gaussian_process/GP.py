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

# GP.py
# Written Ian Rankin - September 2021
#
# A base Gaussian process implementation
# Starting framework for the user preference GP.

import numpy as np
import collections


## get covariance matrix
# calculate the covariance matrix between the samples given in X
# @param X - samples (n1,k) array where n is the number of samples,
#        and k is the dimension of the samples
# @param Y - samples (n2, k)
# @param cov_func - the input covariance function func(u, v, data)
# @param cov_date - any data to pass to the covariance function
#
# @return the covariance matrix of the samples.
def covMatrix(X, Y, cov_func, cov_data):
    cov = np.empty((len(X), len(Y)))

    for i,x1 in enumerate(X):
        for j,x2 in enumerate(Y):
            cov_ij = cov_func(x1, x2, cov_data)
            cov[i,j] = cov_ij
    return cov



## Base Gaussian process class.
#
# This class has the basic functions and API to implement a GP
# Based off of the equations from these helpful blog post
# https://distill.pub/2019/visual-exploration-gaussian-processes/
# http://katbailey.github.io/post/gaussian-processes-for-dummies/
#
# Currently implemented as a k-dimmensional input, 1 output GP.
class GP:
    ## Constructor
    # @param cov_func - the covariance function to use
    # @param cov_data - [opt] any data the covariance function needs.
    # @param mat_inv - [opt] the matrix inversion function to use. By default
    #                   just uses numpy.linalg.inv
    def __init__(self, cov_func, cov_data=None, mat_inv=np.linalg.inv):
        self.cov_func = cov_func
        self.cov_data = cov_data

        self.invert_function = mat_inv
        self.X_train = None
        self.y_train = None


    ## add_training
    # adds training data to the gaussian process
    # appends the data if there already is some training data
    # @param X - the input training data
    # @param y - the output labels of the training data
    # @param training_sigma - [opt] sets the uncertianty in the training data
    #                          accepts scalars or a vector if each sample has
    #                          a different uncertianty.
    def add(self, X, y, training_sigma=0):
        if not isinstance(training_sigma, collections.Sequence):
            training_sigma = np.ones(len(y)) * training_sigma

        if self.X_train is None:
            self.X_train = X
            self.y_train = y
            self.training_sigma = training_sigma
        else:
            self.X_train = np.append(self.X_train, X, axis=0)
            self.y_train = np.append(self.y_train, y, axis=0)
            self.training_sigma = np.append(self.training_sigma, y, axis=0)

    ## clear_training
    # clears all training data from the GP
    def clear(self):
        self.X_train = None
        self.y_train = None

    ## Predicts the output of the GP at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def predict(self, X):
        #### This function treats Y as the training data
        Y = self.X_train
        covXX = covMatrix(X, X, self.cov_func, self.cov_data)
        covXY = covMatrix(X, Y, self.cov_func, self.cov_data)
        covYX = np.transpose(covXY)

        error = np.zeros((len(Y), len(Y)))
        np.fill_diagonal(error, self.training_sigma)

        covYY = covMatrix(Y, Y, self.cov_func, self.cov_data) + error

        covYYinv = self.invert_function(covYY)

        muX_Y = np.matmul(covXY, np.matmul(covYYinv, self.y_train))
        # stored as an instance variable in case it is needed for some reason
        self.cov_predict = covXX -  np.matmul(np.matmul(covXY, covYYinv), covYX)

        sigmaX_Y = np.diagonal(self.cov_predict)
        sigmaX_Y = np.maximum(0, sigmaX_Y)

        return muX_Y, sigmaX_Y

    ## () operator
    # This is just a wrapper around the predict function.
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def __call__(self, X):
        return predict(X)

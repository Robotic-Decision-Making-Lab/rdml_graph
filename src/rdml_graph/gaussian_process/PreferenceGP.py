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

# PreferenceGP.py
# Written Ian Rankin - September 2021
#
# A Gaussian Process implementation that handles ordered pairs of preferences
# for the training data rather than direct absolute samples.
# Essentially optimizes the solution of the samples given to the GP.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence
from rdml_graph.gaussian_process import GP
from rdml_graph.gaussian_process import PreferenceProbit, ProbitBase
from rdml_graph.gaussian_process import k_fold_half, get_dk
from rdml_graph.gaussian_process import PreferenceModel

import scipy.optimize as op
import math

import pdb

#SQ2_Pref_GP = np.sqrt(2)




## PreferenceGP
# A Gaussian Process implementation that handles ordered pairs of preferences
# for the training data rather than direct absolute samples.
# Essentially optimizes the solution of the samples given to the GP.
#
# Based off of the math given in:
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen
class PreferenceGP(GP, PreferenceModel):
    ## constructor
    # @param cov_func - the covariance function to use
    # @param mat_inv - [opt] the matrix inversion function to use. By default
    #                   just uses numpy.linalg.inv
    def __init__(self, cov_func, normalize_gp=True, pareto_pairs=False, \
                normalize_positive=False, other_probits={}, mat_inv=np.linalg.pinv, \
                use_hyper_optimization=False, active_learner=None):
        super(PreferenceGP, self).__init__(cov_func, mat_inv, active_learner=active_learner)
        PreferenceModel.__init__(self, pareto_pairs, other_probits)
        
        self.lambda_gp = 0.1

        self.normalize_gp = normalize_gp
        
        self.normalize_positive = normalize_positive
        self.use_hyper_optimization = use_hyper_optimization

        # sigma on the likelihood function.
        #self.sigma_L = 1.0
        self.probits = [PreferenceProbit(sigma = 1.0)]
        


        for key in other_probits:
            if not isinstance(other_probits[key], ProbitBase):
                raise TypeError("Preference Linear pased a probit that is not a probit: " + str(other_probits[key]))

            self.probits.append(other_probits[key])

        

        self.delta_f = 0.002 # set the convergence to stop
        self.maxloops = 100


    
    ## add
    # adds training data to the gaussian process
    # appends the data if there already is some training data
    # @param X - the input training data
    # @param y - list of discrete pairs [(dk, uk, vk), ...]
    #                       dk = -1 if u > v, dk = 1 if v > u
    #                       uk = index of the input (for the input set of points)
    #                       vk = index of the second input
    #                       @NOTE That this function updates the indicies if there
    #                       is already training data.
    # @param training_sigma - [opt] sets the uncertianty in the training data
    #                          accepts scalars or a vector if each sample has
    #                          a different uncertianty.
    def add(self, X, y, type='relative_discrete', training_sigma=0):
        PreferenceModel.add(self, X,y,type,training_sigma)



    def calc_ll(self):
        K = self.cov_func.cov(self.X_train, self.X_train)
        W, dpy_df, logpYF = self.derivatives(self.y_train, self.F)

        Kinv = self.invert_function(K)
        term2 = 0.5 * np.matmul(np.matmul(np.transpose(self.F), Kinv), self.F)

        term3 = 0.5 * np.log(np.linalg.det(np.identity(len(K)) + np.matmul(K, W)))

        return logpYF - term2 - term3

    def calc_ll_param(self, hyperparameters, X_train, y_train):
        sigma_L = hyperparameters[0]

        self.probits[self.probit_idxs['relative_discrete']].set_sigma(sigma_L)
        self.cov_func.set_param(hyperparameters[1:])
        K = self.cov_func.cov(X_train, X_train)

        W, dpy_df, logpYF = self.derivatives(y_train, self.F)

        Kinv = self.invert_function(K)
        term2 = 0.5 * np.matmul(np.matmul(np.transpose(self.F), Kinv), self.F)

        term3 = 0.5 * np.log(np.linalg.det(np.identity(len(K)) + np.matmul(K, W)))

        #return logpYF
        return logpYF - term2 - term3


    ## optimize_parameters
    # Optimizes the hyperparameters using a generic minimization function
    # This sort of works. In practice I've had issues with it.
    # @param x_train - the input training parameters
    # @param y_train - the label training parameters
    def optimize_parameters(self, x_train, y_train):
        # TODO setup full vector

        #print(self.calc_ll(None))
        #print(self.calc_grad_ll())

        x0 = np.array([self.probits[0].sigma], dtype=np.float)
        x0 = np.append(x0, self.cov_func.get_param(), axis=0)
        x = x0

        ll_pre = self.calc_ll_param(x0, x_train, y_train)
        print('ll_pre = ' + str(ll_pre))

        #bounds=[(0.0001, 100), (0.0001, 10), (0.001,30)],
        bounds = [(0.1, 15) for i in range(len(x))]
        calc_ll = lambda x, *args : -self.calc_ll_param(x, args[0], args[1])
        theta = op.minimize(fun=calc_ll, args=(x_train, y_train), x0=x0, bounds=bounds, tol=0.01, options={'maxiter': 300, 'disp': False})
        x = theta.x
        print(theta)

        self.probits[0].set_sigma(x[0])
        self.cov_func.set_param(x[1:])
        #print('SIGMA_L: ' + str(self.sigma_L))

        ll_post = self.calc_ll_param(x, x_train, y_train)
        print('ll_post = ' + str(ll_post))

    ## derivatives
    # Calculates the derivatives for all of the given probits.
    # @param y - the given set of labels for the probit
    #              this is given as a list of [(dk, u, v), ...]
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of log P(y|x,theta) with respect to F
    #       py - log P(y|x,theta) for the given probit
    def derivatives(self, y, F):
        W = np.zeros((len(F), len(F)))
        grad_ll = np.zeros(len(F))
        log_likelihood = 0

        for j, probit in enumerate(self.probits):
            if y[j] is not None:
                W_local, dpy_df_local, py_local = probit.derivatives(y[j], F)
                try:
                    W += W_local
                except:
                    pdb.set_trace()
                grad_ll += dpy_df_local
                log_likelihood += py_local

        return W, grad_ll, log_likelihood

    ## findMode
    # This function calculates the mode of the F vector by using the
    def findMode(self, x_train, y_train):
        X_train = x_train

        self.K = self.cov_func.cov(X_train, X_train)

        # check convergence using f_error (the max change in the f vector)
        f_error = self.delta_f + 1 # force at least one iteration to be used
        n_loops = 0

        # TODO good way to check for convergence
        while f_error > self.delta_f:
            self.W, self.grad_ll, self.log_likelihood = \
                                            self.derivatives(y_train, self.F)

            F_new = damped_newton_update(
                                        self.F, # estimated training values
                                        self.K, # covariance of training data
                                        self.W, # The W matrix (d2py/ d2df)
                                        self.grad_ll, # The gradient of the log likelihood
                                        self.lambda_gp, # lambda on the newton update
                                        self.invert_function)

            # normalize F
            if self.normalize_gp:
                if self.normalize_positive:
                    min = np.amin(F_new)
                    F_new = (F_new - min)
                    max = np.amax(F_new)
                    F_new = F_new / max
                else:
                    F_norm = np.linalg.norm(F_new, ord=np.inf)
                    F_new = F_new / F_norm

            # check for convergence and add noise if there is an error
            df = np.abs((F_new - self.F))
            if n_loops > 0 and df.max() > f_error:
                print("Laplace error increase, adding noise")
                F_new = F_new + np.random.normal(0, df.max()/10.0, F_new.shape)
            f_error = np.max(df)
            #print('f_error: ' + str(f_error))

            self.F = F_new

            if n_loops % 50 == 0 and n_loops != 0:
                self.F = np.random.random(len(self.X_train))

            n_loops += 1
            if n_loops > self.maxloops:
                print('WARNING: maximum loops in findMode exceeded. Returning current solution')
                break

        # normalize W
        self.W, self.grad_ll, self.log_likelihood = \
                                        self.derivatives(y_train, self.F)


    ## optimize
    # Runs the optimization step required by the user preference GP.
    # @param optimize_hyperparameter - [opt] sets whether to optimize the hyperparameters
    def optimize(self, optimize_hyperparameter=False):
        if optimize_hyperparameter:
            self.F = np.random.random(len(self.X_train))
            split_y = k_fold_half(self.y_train)

            for i in range(len(split_y)):
                train_x = self.X_train
                train_y = split_y[i]

                valid_x = self.X_train
                valid_y = split_y[1-i]

                self.findMode(train_x, train_y)
                self.optimize_parameters(valid_x, valid_y)

        self.F = np.random.random(len(self.X_train))
        self.findMode(self.X_train, self.y_train)

        self.optimized = True





    def predict_large(self,X):
        num_at_a_time = 15

        num_runs = int(math.ceil(X.shape[0] / num_at_a_time))

        mu = np.empty(X.shape[0])
        sigma = np.empty(X.shape[0])

        for i in range(num_runs):
            low_i = i*num_at_a_time
            high_i = min(X.shape[0], low_i+num_at_a_time)

            mu_loc, sigma_loc = self.predict(X[low_i:high_i])
            sigma[low_i:high_i] = sigma_loc
            mu[low_i:high_i] = mu_loc

        return mu, sigma


    ## Predicts the output of the GP at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def predict(self, X):
        if self.X_train is None:
            cov = self.cov_func.cov(X,X)
            sigma = np.diagonal(cov)
            # just in case do to numerical instability a negative variance shows up
            sigma = np.maximum(0, sigma)
            return np.zeros(len(X)), sigma

        # lazy optimization of GP
        if not self.optimized:
            self.optimize(optimize_hyperparameter=self.use_hyper_optimization)

        X_test = X
        X_train = self.X_train
        F = self.F
        K = self.K
        W = self.W

        covXX_test = self.cov_func.cov(X_test, X_train)
        covTestTest = self.cov_func.cov(X_test, X_test)

        covX_testX = np.transpose(covXX_test)

        ####### calculate the mu of the value
        mu = np.matmul(covXX_test, np.matmul(self.invert_function(K), F))


        ######### calculate the covariance and the sigma on the covariance
        tmp = self.invert_function(np.identity(len(K)) + np.matmul(W, K))
        tmp2 = np.matmul(covXX_test, tmp)
        tmp3 = np.matmul(W, covX_testX)

        self.cov = covTestTest - np.matmul(tmp2, tmp3)
        sigma = np.diagonal(self.cov)
        sigma = np.maximum(0, sigma)

        #print(sigma)
        # norm = np.linalg.norm(mu, ord=np.inf)
        # mu = mu / norm
        # sigma = sigma / (norm*norm)


        return mu, sigma



############################# Optimization functions


## damped_newton_update
# this function performs the damped newton update for the preference learning
# @param F - the initial vector of f (estimated training sample outputs)
# @param K - the covariance matrix for training inputs X
# @param W - the W matrix
# @param grad_ll - the gradient of the log_likelihood
# @param lambda - adaptive damping factor on the newton update
# @param invert_function - [opt] the matrix invertion function to use.
#
# @return the updated F for the newton update
def damped_newton_update(F, K, W, grad_ll, lamda, \
                            invert_function=np.linalg.inv):
    term1 = invert_function(K) + W - lamda*np.identity(len(W))
    term1_inv = invert_function(term1)
    term2 = np.matmul((W - lamda*np.identity(len(W))), F) + grad_ll

    return np.matmul(term1_inv, term2)














#

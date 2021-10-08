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
import collections
from rdml_graph.gaussian_process import GP, covMatrix
import scipy.optimize as op
import scipy.stats as st
SQ2_Pref_GP = np.sqrt(2)



def get_dk(u, v):
    if u > v:
        return -1
    elif u < v:
        return 1
    else:
        return -1 # probably handle it this way... I could also probably just return 0

## generate_fake_pairs
# generates a set of pairs of data from faked data
# helper function for fake input data
# @param X - the inputs to the function
# @param real_f - the real function to estimate
def generate_fake_pairs(X, real_f, pair_i, data=None):
    Y = real_f(X, data=data)

    pairs = [(get_dk(Y[pair_i], y),pair_i, i) for i, y in enumerate(Y)]
    return pairs

## PreferenceGP
# A Gaussian Process implementation that handles ordered pairs of preferences
# for the training data rather than direct absolute samples.
# Essentially optimizes the solution of the samples given to the GP.
#
# Based off of the math given in:
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen
class PreferenceGP(GP):
    ## constructor
    # @param cov_func - the covariance function to use
    # @param cov_data - any data the covariance function needs.
    # @param mat_inv - [opt] the matrix inversion function to use. By default
    #                   just uses numpy.linalg.inv
    def __init__(self, cov_func, cov_data, mat_inv=np.linalg.inv):
        super(PreferenceGP, self).__init__(cov_func, cov_data, mat_inv)

        self.optimized = False
        self.lambda_gp = 0.3
        # sigma on the likelihood function.
        self.sigma_L = 0.3


    ## add_training
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
    def add(self, X, y, training_sigma=0):
        if not isinstance(training_sigma, collections.Sequence):
            training_sigma = np.ones(len(y)) * training_sigma

        if self.X_train is None:
            self.X_train = X
            self.y_train = y
            self.training_sigma = training_sigma
        else:
            # reset index
            len_X = len(self.X_train)
            y = [(d, u+len_X, v+len_X) for d, u, v in y]

            self.X_train = np.append(self.X_train, X, axis=0)
            self.y_train = self.y_train + y
            self.training_sigma = np.append(self.training_sigma, y, axis=0)
        self.optimized = False




    def optimize_parameters(self):
        # TODO setup full vector

        dsigmaL = calc_evidence_derivative_likelihood(
                                self.y_train,
                                self.F,
                                self.W,
                                self.K,
                                self.sigma_L,
                                self.invert_function)

        theta = [self.sigma_L]
        dllTheta = [dsigmaL]




    ## optimize
    # Runs the optimization step required by the user preference GP.
    def optimize(self):
        # initial F estimate
        self.F = np.random.random(len(self.X_train))
        X_train = self.X_train
        pairs = self.y_train

        self.K = covMatrix(X_train, X_train, self.cov_func, self.cov_data)

        # TODO good way to check for convergence
        for i in range(20):
            self.F, self.W = damped_newton_update(
                                        self.y_train, # input training pairs
                                        self.F, # estimated training values
                                        self.K, # covariance of training data
                                        self.sigma_L, # sigma on the liklihood function
                                        self.lambda_gp, # lambda on the newton update
                                        self.invert_function)
            #self.optimize_parameters()


        self.optimized = True






    ## Predicts the output of the GP at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def predict(self, X):
        # lazy optimization of GP
        if not self.optimized:
            self.optimize()

        W = self.W
        X_test = X
        X_train = self.X_train
        F = self.F
        K = self.K
        W = self.W

        covXX_test = covMatrix(X_test, X_train, self.cov_func, self.cov_data)
        covTestTest = covMatrix(X_test, X_test, self.cov_func, self.cov_data)

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

        return mu, sigma






## derv_discrete_loglike
# Calculates the first derivative of log likelihood.
# Appendix A.1.1.1.1
# Assumes (f(vk), f(uk))
#
# xi, yk, vk, are indicies of the likelihood
# @param F - the vector of F (estimated training sample outputs)
# @param dk - the label for the given sample of uk, vk
# @param xi - the index of f
# @param uk - index of the u parameters of the ordered pair
# @param vk - index of the v parameters of the ordered pair
# @param sigma - the sigma attached to the probid for discrete
def derv_discrete_loglike(F, dk, xi, uk, vk, sigma):
    if xi == uk:
        I = 1
    elif xi == vk:
        I = -1
    else:
        return 0

    zk = relative_probit(F, dk, uk, vk, sigma)
    return I * dk * (1 / st.norm.pdf(zk)) * (1 / (SQ2_Pref_GP * sigma)) * st.norm.cdf(zk)


## derv_param_discrete_loglike
# Calculate the derivative of the discrete log liklihood with respect to the parameters
# d / d(sigma) log p(d | f(u), f(v))
#
# xi, yk, vk, are indicies of the likelihood
# @param F - the vector of F (estimated training sample outputs)
# @param dk - the label for the given sample of uk, vk
# @param xi - the index of f
# @param uk - index of the u parameters of the ordered pair
# @param vk - index of the v parameters of the ordered pair
# @param sigma - the sigma attached to the probid for discrete
def derv_param_discrete_loglike(F, dk, xi, uk, vk, sigma):
    zk = relative_probit(F, dk, uk, vk, sigma)

    # calculate the derivative
    pdf_zk = st.norm.pdf(zk)
    cdf_zk = st.norm.cdf(zk)

    return -dk * pdf_zk / cdf_zk

## relative_probit
# calculates the probit for the relative likelyhood function
# Lrel
# @param F - the vector of F (estimated training sample outputs)
# @param d - the label of the ordered pair
# @param u - the index of the u element
# @param v - the index of the v element
# @param sigma - the sigma attached to he relative likelyhood function.
def relative_probit(F, d, u, v, sigma):
    return (d * (F[u] - F[v])) / (SQ2_Pref_GP * sigma)


## derv2_discrete_loglike
# Calculates the second derivative of discrete log likelihood.
#  d / (dF(xi)dF(xj)) ln(p(dk|F(u), F(v)))
#
# Appendix A.1.1.1
# Assumes (f(vk), f(uk))
# xi, yk, vk, are indicies of the likelihood
# @param F - the vector of f (estimated training sample outputs)
# @param dk - the label for the given sample of uk, vk
# @param xi - the index of f
# @param xj - the second index of f
# @param uk - index of the u parameters of the ordered pair
# @param vk - index of the v parameters of the ordered pair
# @param sigma - the sigma attached to the probid for discrete
def derv2_discrete_loglike(F, dk, xi, xj, uk, vk, sigma):
    # setup the indicator variables.
    if xi == uk:
        I1 = 1
    elif xi == vk:
        I1 = -1
    else:
        return 0

    if xj == uk:
        I2 = 1
    elif xj == vk:
        I2 = -1
    else:
        return 0

    zk = relative_probit(F, dk, uk, vk, sigma)

    # calculate the derivative
    pdf_zk = st.norm.pdf(zk)
    cdf_zk = st.norm.cdf(zk)

    paren = (zk * pdf_zk / cdf_zk) + ((pdf_zk*pdf_zk) / (cdf_zk*cdf_zk))
    return -(dk*dk)*I1*I2*(1/(2*sigma*sigma)) * paren

## derv_W_discrete_sigma
# Calculates the derivative of the of the second derivative of the discrete
# log likelihood with respect to the sigma parameter
# dW / d(sigma) for the discrete W
# @param F - the vector of f (estimated training sample outputs)
# @param dk - the label for the given sample of uk, vk
# @param xi - the index of f
# @param xj - the second index of f
# @param uk - index of the u parameters of the ordered pair
# @param vk - index of the v parameters of the ordered pair
# @param sigma - the sigma attached to the probid for discrete
def derv_W_discrete_sigma(F, dk, xi, xj, uk, vk, sigma):
    # setup the indicator variables.
    if xi == uk:
        I1 = 1
    elif xi == vk:
        I1 = -1
    else:
        return 0

    if xj == uk:
        I2 = 1
    elif xj == vk:
        I2 = -1
    else:
        return 0

    zk = relative_probit(F, dk, uk, vk, sigma)

    pdf_zk = st.norm.pdf(zk)
    cdf_zk = st.norm.cdf(zk)


    first_term = (dk*dk/(sigma**3))*((dk*pdf_zk / cdf_zk) + ((pdf_zk**2) / (cdf_zk**2)))
    tmp = (-(cdf_zk**2) + (dk**2)*(cdf_zk**2) + 3*dk*pdf_zk*cdf_zk + 2*(pdf_zk**2))
    tmp = tmp / (cdf_zk**3)
    second_term = (1 / (2*sigma*sigma*sigma)) * dk*dk*zk*pdf_zk * tmp

    return -I1*I2*(first_term - second_term)




## calc_W_discrete
# calculate the W matrix for the discrete relative pairs contribution
# This is following the formulation given in Appendix B.1.1.6
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
#
# @return W (n,n) n is the length of F
def calc_W_discrete(pairs, F, sigma_L):
    W = np.zeros((len(F), len(F)))

    for dk, uk, vk in pairs:
        W[uk, uk] -= derv2_discrete_loglike(F, dk, xi=uk, xj=uk, uk=uk, vk=vk, sigma=sigma_L)
        W[uk, vk] -= derv2_discrete_loglike(F, dk, xi=uk, xj=vk, uk=uk, vk=vk, sigma=sigma_L)
        W[vk, uk] -= derv2_discrete_loglike(F, dk, xi=vk, xj=uk, uk=uk, vk=vk, sigma=sigma_L)
        W[vk, vk] -= derv2_discrete_loglike(F, dk, xi=vk, xj=vk, uk=uk, vk=vk, sigma=sigma_L)

    return W

## calc_W_discrete_derv_param
# Calculates the derivative of the of the second derivative of the discrete
# log likelihood with respect to the sigma parameter
# dW / d(sigma) for the discrete W
# This is following the formulation given in Appendix B.2.3
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
#
# @return W (n,n) n is the length of F
def calc_W_discrete_derv_param(pairs, F, sigma_L):
    dW = np.zeros((len(F), len(F)))

    for dk, uk, vk in pairs:
        dW[uk, uk] += derv_W_discrete_sigma(F, dk, xi=uk, xj=uk, uk=uk, vk=vk, sigma=sigma_L)
        dW[uk, vk] += derv_W_discrete_sigma(F, dk, xi=uk, xj=vk, uk=uk, vk=vk, sigma=sigma_L)
        dW[vk, uk] += derv_W_discrete_sigma(F, dk, xi=vk, xj=uk, uk=uk, vk=vk, sigma=sigma_L)
        dW[vk, vk] += derv_W_discrete_sigma(F, dk, xi=vk, xj=vk, uk=uk, vk=vk, sigma=sigma_L)

    return dW


## calc_evidence_derivative_likelihood
# calculates the derivative of the evidence derivatives
# d p(theta | Y, X) / dTheta
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param K - the covariance matrix
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
# @param invert_function - [opt] the matrix invertion function to use.
#
# @return the derivative of the log liklihood with respect to the liklihood parameter
def calc_evidence_derivative_likelihood(pairs, F, W, K, sigma_L, invert_function=np.linalg.inv):
    term1 = calc_grad_loglike_discrete_param(pairs, F, sigma_L)
    partialW_sigma = calc_W_discrete_derv_param(pairs, F, sigma_L)

    tmp1 = invert_function(np.identity(len(K)) + np.matmul(K, W))
    term2 = -0.5 * np.trace(np.matmul(tmp1, np.matmul(K, partialW_sigma) ) )

    return term1 + term2

## calc_grad_loglike_discrete
# Calculate the gradient of the log-likelyhood
# this is used in the update step
# grad log p(Y|F,thetaL)
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
#
# @return grad log p(Y|F,thetaL) (n,) n is the length of F
def calc_grad_loglike_discrete(pairs, F, sigma_L):
    grad = np.zeros((len(F),))

    for dk, uk, vk in pairs:
        grad[uk] -= derv_discrete_loglike(F, dk, xi=uk, uk=uk, vk=vk, sigma=sigma_L)
        grad[vk] -= derv_discrete_loglike(F, dk, xi=vk, uk=uk, vk=vk, sigma=sigma_L)

    return grad

## calc_grad_loglike_discrete_param
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
#
# @return grad log p(Y|F,thetaL) (n,) n is the length of F
def calc_grad_loglike_discrete_param(pairs, F, sigma_L):
    grad = np.zeros((len(F),))

    for dk, uk, vk in pairs:
        grad[uk] += derv_param_discrete_loglike(F, dk, xi=uk, uk=uk, vk=vk, sigma=sigma_L)
        grad[vk] += derv_param_discrete_loglike(F, dk, xi=vk, uk=uk, vk=vk, sigma=sigma_L)

    return grad


## damped_newton_update
# this function performs the damped newton update for the preference learning
# @param pairs_discrete - the pairs the list of ordered pairs (dk, uk, vk)
#                         where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the initial vector of f (estimated training sample outputs)
# @param K - the covariance matrix for training inputs X
# @param sigma_L - the sigma for the likelyhood of pairs
# @param lambda - adaptive damping factor on the newton update
# @param invert_function - [opt] the matrix invertion function to use.
#
# @return the updated F for the newton update
def damped_newton_update(pairs_discrete, F, K, sigma_L, lamda, \
                            invert_function=np.linalg.inv):
    W = calc_W_discrete(pairs_discrete, F, sigma_L = sigma_L)
    grad_ll = calc_grad_loglike_discrete(pairs_discrete, F, sigma_L=1)

    term1 = invert_function(K) + W - lamda*np.identity(len(W))
    term1_inv = invert_function(term1)
    term2 = np.matmul((W - lamda*np.identity(len(W))), F) + grad_ll

    return np.matmul(term1_inv, term2), W
















#

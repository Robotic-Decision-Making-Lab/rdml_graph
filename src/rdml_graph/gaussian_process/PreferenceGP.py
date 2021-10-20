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

import pdb

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
    # @param mat_inv - [opt] the matrix inversion function to use. By default
    #                   just uses numpy.linalg.inv
    def __init__(self, cov_func, mat_inv=np.linalg.inv):
        super(PreferenceGP, self).__init__(cov_func, mat_inv)

        self.optimized = False
        self.lambda_gp = 0.3
        # sigma on the likelihood function.
        self.sigma_L = 1.0
        #self.sigma_L = 0.005



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
    def add(self, X, y, type='relative_discrete', training_sigma=0):
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




    def calc_ll(self, x, *args):
        sigma_L = x[0]
        X_train = args[0]
        y_train = args[1]

        self.cov_func.set_param(x[1:])
        K = self.cov_func.cov(X_train, X_train)
        W = calc_W_discrete(y_train, self.F, sigma_L)
        return -logliklihoodYXTh(y_train,
                                self.F,
                                W,
                                K,
                                sigma_L,
                                self.invert_function)

    def calc_grad_ll(self, x, *args):
        X_train = args[0]
        y_train = args[1]

        sigma_L = x[0]

        #x[2] = self.cov_func.get_param[1]
        self.cov_func.set_param(x[1:])
        K = self.cov_func.cov(X_train, X_train)
        W = calc_W_discrete(y_train, self.F, sigma_L = sigma_L)
        dSigmaL = calc_evidence_derivative_likelihood(
                                y_train,
                                self.F,
                                W,
                                K,
                                sigma_L,
                                self.invert_function)


        grad = np.empty(len(self.cov_func))
        dk_all = self.cov_func.cov_gradient(X_train, X_train)
        for k in range(len(self.cov_func)):
            dK = dk_all[:,:,k]
            grad[k] = derv_log_p_y_given_theta_for_theta(
                                        y_train,
                                        sigma_L,
                                        self.F,
                                        K,
                                        dK,
                                        W, \
                                    self.invert_function)

        liklihood_theta = np.array([dSigmaL])
        theta = np.append(liklihood_theta, grad, axis=0)
        #theta = liklihood_theta

        return -theta

    def optimize_parameters(self, x_train, y_train):
        # TODO setup full vector

        #print(self.calc_ll(None))
        #print(self.calc_grad_ll())

        x0 = np.array([self.sigma_L], dtype=np.float)
        x0 = np.append(x0, self.cov_func.get_param(), axis=0)
        x = x0

        #pdb.set_trace()

        # for i in range(40):
        #     print('STEP: '+str(i))
        #     grad = self.calc_grad_ll(x)
        #     value = self.calc_ll(x)
        #
        #
        #     print(grad)
        #     print(value)
        #     print(x)
        #
        #     x -= grad*0.02
        #     #x[0] = self.sigma_L
        #     # if x[0] <= 0.01:
        #     #     x[0] = 0.01

        #theta = op.minimize(fun=self.calc_ll, args=(x_train, y_train), x0=x0, jac=self.calc_grad_ll, method='BFGS', options={'maxiter': 5, 'disp': True})

        ll_pre = self.calc_ll(x0, x_train, y_train)
        #print(ll_pre)

        #bounds=[(0.0001, 100), (0.0001, 10), (0.001,30)],
        bounds = [(0.1, 20) for i in range(len(x))]
        theta = op.minimize(fun=self.calc_ll, args=(x_train, y_train), x0=x0, bounds=bounds, tol=0.01, options={'maxiter': 300, 'disp': False})
        x = theta.x
        print(theta)

        self.sigma_L = x[0]
        self.cov_func.set_param(x[1:])
        #print('SIGMA_L: ' + str(self.sigma_L))

        ll_post = self.calc_ll(x, x_train, y_train)
        #print(ll_post)

        #pdb.set_trace()



    def findMode(self, x_train, y_train):
        # initial F estimate
        #self.F = np.random.random(len(self.X_train))
        X_train = x_train
        #pairs = y_train

        self.K = covMatrix(X_train, X_train, self.cov_func)

        # split dataset


        # TODO good way to check for convergence
        for i in range(10):
            self.F, self.W = damped_newton_update(
                                        y_train, # input training pairs
                                        self.F, # estimated training values
                                        self.K, # covariance of training data
                                        self.sigma_L, # sigma on the liklihood function
                                        self.lambda_gp, # lambda on the newton update
                                        self.invert_function)
            # normalize F
            self.F = self.F / np.linalg.norm(self.F, ord=np.inf)


    ## optimize
    # Runs the optimization step required by the user preference GP.
    # @param optimize_hyperparameter - [opt] sets whether to optimize the hyperparameters
    def optimize(self, optimize_hyperparameter=False):
        self.F = np.random.random(len(self.X_train))

        if optimize_hyperparameter:
            shuffle = np.arange(len(self.y_train))
            np.random.shuffle(shuffle)

            self.k_fold = 2
            splits = np.array_split(shuffle, self.k_fold)


            for i in range(self.k_fold):
                # if i != 0:
                #     print('Optimize with find mode before and after:')
                #     t1 = self.calc_ll([self.sigma_L])
                train_x = self.X_train

                train_split = splits[0:i] + splits[(i+1):]
                t_split = np.empty(0, dtype=np.int)
                for s in train_split:
                    t_split = np.append(t_split, s)

                train_y = [self.y_train[idx] for idx in t_split]

                valid_x = self.X_train
                valid_y = [self.y_train[idx] for idx in splits[i]]

                self.findMode(train_x, train_y)
                #self.findMode(self.X_train, self.y_train)
                # if i != 0:
                #     t2 = self.calc_ll([self.sigma_L])
                #     print(t1)
                #     print(t2)
                self.optimize_parameters(valid_x, valid_y)

        self.findMode(self.X_train, self.y_train)

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

        covXX_test = covMatrix(X_test, X_train, self.cov_func)
        covTestTest = covMatrix(X_test, X_test, self.cov_func)

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




## relative_probit
# calculates the probit for the relative likelyhood function
# Lrel
# @param F - the vector of F (estimated training sample outputs)
# @param d - the label of the ordered pair
# @param u - the index of the u element
# @param v - the index of the v element
# @param sigma - the sigma attached to he relative likelyhood function.
def relative_probit(F, d, u, v, sigma):
    return (d * (F[v] - F[u])) / (SQ2_Pref_GP * sigma)

def calc_pdf_o_cdf(pdf_zk, cdf_zk):
    # as zk -> -infinity then pdf_zk / cdf_zk goes to infinity
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D%2Cx-%3E-%E2%88%9E%5D
    # As zk -> infinity then pdf_zk / cdf_zk goes to 0
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D%2Cx-%3E%E2%88%9E%5D
    # this code checks for those states and inputs the appropriate values
    #

    if cdf_zk == 0:
        if pdf_zk < 0.5:
            pdf_cdf_zk = pdf_cdf_2 = float('inf')
        else:
            pdf_cdf_zk = pdf_cdf_2 = 0
    else:
        pdf_cdf_zk = pdf_zk / cdf_zk
        pdf_cdf_2 = pdf_cdf_zk * pdf_cdf_zk

    return pdf_cdf_zk, pdf_cdf_2

## derv2_discrete_loglike
# Calculates the second derivative of discrete log likelihood.
#  d / (dF(xi)dF(xj)) ln(p(dk|F(u), F(v)))
#
# Appendix A.1.1.2
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

    pdf_cdf_zk, pdf_cdf_2 = calc_pdf_o_cdf(pdf_zk, cdf_zk)

    # when zk -> -infinity there is a -infinity + infinity limit for paren
    # using wolfram this comes out as -infinity
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5Bx*Divide%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2Cerfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%5D%2Cx-%3E-%E2%88%9E%5D%2BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D
    # This is handled in this code
    # EDIT: Hmm, 0 seems to make it more stable... Not sure why, but it seems to
    # be working.
    if cdf_zk == 0 and zk < 0:
        #paren = -float('inf')
        paren = 0
    else:
        paren = (zk * pdf_cdf_zk) + (pdf_cdf_2)

    w_ij = -(dk*dk)*I1*I2*(1/(2*sigma*sigma)) * paren

    if np.isnan(w_ij):
        pdb.set_trace()

    return w_ij

## derv2_discrete_loglike
# Calculates the third derivative of discrete log likelihood.
#  d / (dF(xi)dF(xj)F(xi)) ln(p(dk|F(u), F(v)))
#
# Appendix A.1.1.1
# Assumes (f(vk), f(uk))
# xi, yk, vk, are indicies of the likelihood
# @param F - the vector of f (estimated training sample outputs)
# @param dk - the label for the given sample of uk, vk
# @param xi - the index of f
# @param xj - the second index of f
# @param xk - the third index to differentiate with resepect to.
# @param uk - index of the u parameters of the ordered pair
# @param vk - index of the v parameters of the ordered pair
# @param sigma - the sigma attached to the probid for discrete
def derv3_discrete_loglike(F, dk, xi, xj, xk, uk, vk, sigma):
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

    if xk == uk:
        I3 = 1
    elif xk == vk:
        I3 = -1
    else:
        return 0

    zk = relative_probit(F, dk, uk, vk, sigma)
    pdf_zk = st.norm.pdf(zk)
    cdf_zk = st.norm.cdf(zk)

    pdf_cdf_zk, pdf_cdf_2 = calc_pdf_o_cdf(pdf_zk, cdf_zk)


    # when zk -> -infinity there is a -infinity + infinity limit for paren
    # using wolfram this comes out as -infinity
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5Bx*Divide%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2Cerfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%5D%2Cx-%3E-%E2%88%9E%5D%2BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D
    # This is handled in this code
    # EDIT: Hmm, 0 seems to make it more stable... Not sure why, but it seems to
    # be working.
    if cdf_zk == 0 and zk < 0:
        #paren = -float('inf')
        paren = 0
    else:
        paren = pdf_cdf_zk - (dk*dk*pdf_cdf_zk) - (3*dk*pdf_cdf_2) - (2 * pdf_cdf_2 * pdf_zk)

    return -(dk*dk*dk / (2*SQ2_Pref_GP*sigma*sigma*sigma))*I1*I2*I3*paren





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

## calc_derv_W_discrete
# calculate the derivative W matrix for the discrete relative pairs contribution with respect to f_i
# dW / dF(i)
#
# @param i - the index of the F vector that is being calculated
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
#
# @return W (n,n) n is the length of F
def calc_derv_W_discrete( i, pairs, F, sigma_L):
    W = np.zeros((len(F), len(F)))

    for dk, uk, vk in pairs:
        W[uk, uk] -= derv3_discrete_loglike(F, dk, xi=uk, xj=uk, xk=i, uk=uk, vk=vk, sigma=sigma_L)
        W[uk, vk] -= derv3_discrete_loglike(F, dk, xi=uk, xj=vk, xk=i, uk=uk, vk=vk, sigma=sigma_L)
        W[vk, uk] -= derv3_discrete_loglike(F, dk, xi=vk, xj=uk, xk=i, uk=uk, vk=vk, sigma=sigma_L)
        W[vk, vk] -= derv3_discrete_loglike(F, dk, xi=vk, xj=vk, xk=i, uk=uk, vk=vk, sigma=sigma_L)

    return W













## logliklihoodYXTh
# calculates the log-likilihood conditioned on X and theta
# p(Y|X,Theta)
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param W - The W matrix
# @param K - the covariance matrix
def logliklihoodYXTh(pairs, F, W, K, sigmaL, invert_function=np.linalg.inv):
    logpYF = 0
    for dk, uk, vk in pairs:
        logpYF += np.log(st.norm.cdf(relative_probit(F, dk, uk, vk, sigmaL)))

    Kinv = invert_function(K)
    term2 = 0.5 * np.matmul(np.matmul(np.transpose(F), Kinv), F)

    term3 = 0.5 * np.log(np.linalg.det(np.identity(len(K)) + np.matmul(K, W)))

    return logpYF - term2 - term3

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


################################## calculate gradient of parameters (evidence)


#### Discrete relative



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

    return (term1 + term2)

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


    pdf_cdf_zk, pdf_cdf_2 = calc_pdf_o_cdf(pdf_zk, cdf_zk)
    pdf_cdf_3 = pdf_cdf_2*pdf_cdf_zk


    tmp = (-(cdf_zk**2) + (dk**2)*(cdf_zk**2) + 3*dk*pdf_zk*cdf_zk + 2*(pdf_zk**2))
    if cdf_zk == 0 and zk < 0:
        #not sure this makes sense, but I think setting it to zero is the best option.
        tmp = 0
    else:
        tmp = -pdf_cdf_zk + zk*zk*pdf_cdf_zk + 3*zk*pdf_cdf_2 + 2*pdf_cdf_3
        first_term = (dk*dk/(sigma**3))*((dk*pdf_cdf_zk) + pdf_cdf_2)

    second_term = (1 / (2*sigma*sigma*sigma)) * dk*dk*zk * tmp

    return -I1*I2*(first_term - second_term)

## calc_grad_loglike_discrete_param
# @param pairs - the list of ordered pairs (dk, uk, vk) where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param F - the vector of f (estimated training sample outputs)
# @param sigma_L - the sigma on the likelyhood function (hyperparameter)
#
# @return grad log p(Y|F,thetaL) (n,) n is the length of F
def calc_grad_loglike_discrete_param(pairs, F, sigma_L):
    grad = 0

    for dk, uk, vk in pairs:
        grad += derv_param_discrete_loglike(F, dk, xi=uk, uk=uk, vk=vk, sigma=sigma_L)
        grad += derv_param_discrete_loglike(F, dk, xi=vk, uk=uk, vk=vk, sigma=sigma_L)

    return grad




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


################################## calculate gradient of parameters (covariance function)

## derv_log_p_y_given_theta_for_theta
# caluclate the derivative of the log liklihood for a given parameter in the
# covariance function
# d log p(Y|theta)/dtheta_j
#
# @param pairs_discrete - the pairs the list of ordered pairs (dk, uk, vk)
#                         where dk=-1 indicates u > v, and dk=1 indicate y < v
# @param sigma_L - the sigma for the likelyhood of pairs
# @param F - the vector of f (estimated training sample outputs)
# @param K - the covariance matrix
# @param dK - the derivative of the covariance matrix for the given parameter
# @param W - The W matrix
# @param invert_function - [opt] the matrix invertion function to use.
#
# @return the derivative of the log liklihood with respect to the parameter
def derv_log_p_y_given_theta_for_theta(pairs_discrete, sigma_L, F, K, dK, W, \
                        invert_function=np.linalg.inv):
    K_inv = invert_function(K)
    termA_1 = 0.5 * np.matmul(np.matmul(np.transpose(F), K_inv), np.matmul(dK, np.matmul(K_inv, F)))
    tmp = np.identity(len(W)) + np.matmul(K, W)
    tmp = invert_function(tmp)
    termA_2 = 0.5 * np.trace(np.matmul(tmp, np.matmul(dK, W)))
    termA = termA_1 - termA_2

    tmp = invert_function(np.identity(len(W)) + np.matmul(K, W))
    grad_ll = calc_grad_loglike_discrete(pairs_discrete, F, sigma_L)
    factor2 = np.matmul(tmp, np.matmul(dK, grad_ll))


    termB = 0
    for i in range(len(F)):
        gradW_i = calc_derv_W_discrete(i, pairs_discrete, F, sigma_L)
        factor1 = -0.5 * np.trace(np.matmul(tmp, np.matmul(K, gradW_i)))

        termB += factor1 * factor2[i]

    return termA + termB


############################# Optimization functions

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
    #print('W:')
    #print(W)
    grad_ll = calc_grad_loglike_discrete(pairs_discrete, F, sigma_L=sigma_L)

    #print('grad_ll')
    #print(grad_ll)

    term1 = invert_function(K) + W - lamda*np.identity(len(W))
    term1_inv = invert_function(term1)
    term2 = np.matmul((W - lamda*np.identity(len(W))), F) + grad_ll

    return np.matmul(term1_inv, term2), W
















#

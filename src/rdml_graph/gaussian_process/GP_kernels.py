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

# GP_kernels.py
# Written Ian Rankin - September 2021
#
# A set of kernels for gaussian processes.

import numpy as np
import collections
import pdb

## Base kernel function class
class kernel_func:
    def __init__(self):
        pass

    ## get covariance matrix
    # calculate the covariance matrix between the samples given in X
    # @param X - samples (n1,k) array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k)
    #
    # @return the covariance matrix of the samples.
    def cov(self, X, Y):
        cov = np.empty((len(X), len(Y)))

        for i,x1 in enumerate(X):
            for j,x2 in enumerate(Y):
                cov_ij = self.__call__(x1, x2)
                cov[i,j] = cov_ij
        return cov

    ## get gradient of the covariance matrix
    # calculate the covariance matrix between the samples given in X
    # @param X - samples (n1,k) array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k)
    #
    # @return the covariance gradient tensor of the samples. [n1, n2, k]
    def cov_gradient(self, X, Y):
        cov = np.empty((len(X), len(Y), len(self)))

        for i,x1 in enumerate(X):
            for j,x2 in enumerate(Y):
                cov_ij = self.gradient(x1, x2)
                cov[i,j, :] = cov_ij
        return cov

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        raise NotImplementedError('kernel_func update function not implemented')

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        return np.empty(0)

    def gradient(self, u, v):
        raise NotImplementedError('kernel_func gradient function not implemented')

    def __call__(self, u, v):
        raise NotImplementedError('kernel_func __call__ function not implemented')

    # self + other
    def __add__(self, other):
        if isinstance(other, kernel_func):
            return dual_kern(self, other, '+')
        else:
            raise TypeError('kernel function add passed a type ' + str(type(other)))

    # self * other
    def __mul__(self, other):
        if isinstance(other, kernel_func):
            return dual_kern(self, other, '*')
        else:
            raise TypeError('kernel function multiply passed a type ' + str(type(other)))

    def __len__(self):
        return 0

## kernel function class to handle adding or multiplying
# kernel functions together.
# allows stacking of kernel function such as.
# (gr.RBF_kern(1,1) * gr.periodic_kern(1,1,1)) + gr.linear_kern(1,1,1)
class dual_kern(kernel_func):
    ## Constructor
    # @param kern_1 - the first kernel_func
    # @param kern_2 - the second kernel_func
    # @param operator - the operator to apply between the two kernel function
    #           supports ['+', '*']
    def __init__(self, kern_1, kern_2, operator='+'):
        super(dual_kern, self).__init__()
        self.a = kern_1
        self.b = kern_2

        self.operator = operator

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        if isinstance(theta, collections.Sequence) and not isinstance(theta, np.ndarray):
            theta = np.array(theta)

        self.a.set_param(theta[:len(self.a)])
        self.b.set_param(theta[len(self.a):])

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        a_param = self.a.get_param()
        b_param = self.b.get_param()

        return np.append(a_param, b_param, axis=0)

    def gradient(self, u, v):
        if self.operator == '+':
            a_grad = self.a.gradient(u,v)
            b_grad = self.b.gradient(u,v)
        elif self.operator == '*':
            a_grad = self.a.gradient(u,v) * self.b(u,v)
            b_grad = self.b.gradient(u,v) * self.a(u,v)
        else:
            raise NotImplementedError('dual_kern does not have operator `'+self.operator+'` implemented')

        return np.append(a_grad, b_grad, axis=0)

    def __call__(self, u, v):
        a_f = self.a(u,v)
        b_f = self.b(u,v)

        if self.operator == '+':
            return a_f + b_f
        elif self.operator == '*':
            return a_f * b_f
        else:
            raise NotImplementedError('dual_kern does not have operator `'+self.operator+'` implemented')

    def __len__(self):
        return len(self.a)+len(self.b)





## RBF_kern
# Radial basis function for two points.
# This is a single input sample (v (k) and u (k))
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class RBF_kern(kernel_func):

    ## Constructor
    # @param sigma - the sigma for the rbf kernel
    # @param l - the lengthscale for the rbf_kernel
    def __init__(self, sigma, l):
        super(RBF_kern, self).__init__()

        self.sigma = sigma
        self.l = l

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        self.sigma = theta[0]
        self.l = theta[1]

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        theta = np.array([self.sigma, self.l])
        return theta

    def gradient(self, u, v):
        top = (u-v)
        top = np.sum(top*top)

        exp_x = np.exp(-top / (2 * self.l*self.l))

        dSigma = 2 * self.sigma * exp_x
        dl = 2 * self.sigma * self.sigma * top * exp_x / (self.l*self.l*self.l)

        return np.array([dSigma, dl])

    def __call__(self, u, v):
        top = (u - v)
        top = np.sum(top * top)

        return self.sigma*self.sigma * np.exp(-top / (2 * self.l*self.l))

    def __len__(self):
        return 2

## periodic_kern
# A periodic kernel for gaussian processes
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class periodic_kern(kernel_func):

    ## Constructor
    # @param sigma - the sigma for the rbf kernel
    # @param - sigma, the sigma for the periodic kernel
    # @param - l, the lengthscale for the periodic kernel
    # @param - p, the periodicity of the periodic kernel
    def __init__(self, sigma, l, p):
        super(periodic_kern, self).__init__()

        self.sigma = sigma
        self.l = l
        self.p = p

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        self.sigma = theta[0]
        self.l = theta[1]
        self.p = theta[2]

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        theta = np.array([self.sigma, self.l, self.p])
        return theta

    def gradient(self, u, v):
        uv_norm = np.sum(np.abs(u-v)) #np.linalg.norm(u-v, ord=1)
        sin_tmp = np.sin(np.pi*uv_norm / self.p)
        exp_int = - 2 * sin_tmp * sin_tmp / (self.l*self.l)
        exp_x = np.exp(exp_int)

        dSigma = 2 * self.sigma * exp_x

        dl = -2 * self.sigma*self.sigma * exp_x * uv_norm / (self.l*self.l*self.l)

        dp = 2 * np.pi * self.sigma * self.sigma * uv_norm * exp_x * \
            np.cos(np.pi * uv_norm / self.p) / (self.l*self.l * self.p*self.p)

        return np.array([dSigma, dl, dp])

    def __call__(self, u, v):
        uv_norm = np.sum(np.abs(u-v)) #np.linalg.norm(u-v, ord=1)
        sin_tmp = np.sin(np.pi*uv_norm / self.p)
        exp_int = - 2 * sin_tmp * sin_tmp / (self.l*self.l)

        return self.sigma*self.sigma * np.exp(exp_int)

    def __len__(self):
        return 3



## linear_kern
# A linear kernel for gaussian processes
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class linear_kern(kernel_func):

    ## Constructor
    # @param sigma - the sigma for the linear kernel
    # @param - sigma, the sigma for the linear kernel
    # @param - sigma_b, the sigma_b for the linear
    # @param - c, the offset for the linear kernel
    def __init__(self, sigma, sigma_b, c):
        super(linear_kern, self).__init__()

        self.sigma = sigma
        self.sigma_b = sigma_b
        self.c = c

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        self.sigma = theta[0]
        self.sigma_b = theta[1]
        self.c = theta[2]

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        theta = np.array([self.sigma, self.sigma_b, self.c])
        return theta

    def gradient(self, u, v):
        dSigma_b = 2 * self.sigma_b

        dSigma = np.sum(2*self.sigma*(u-self.c)*(v-self.c))
        dc = self.sigma*self.sigma*np.sum(2*self.c - u - v)

        return np.array([dSigma, dSigma_b, dc])

    def __call__(self, u, v):
        return (self.sigma_b**2) + ((self.sigma**2) * np.sum((u-self.c) * (v-self.c)))

    def __len__(self):
        return 3

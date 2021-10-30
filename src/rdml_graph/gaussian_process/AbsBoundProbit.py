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

# OrdinalProbit.py
# Written Ian Rankin - October 2021
# Modified code by Nicholas Lawerence from here
# https://github.com/osurdml/GPtest/tree/feature/wine_statruns/gp_tools
# Used with permission.
#
# A set of different Gaussian Process Probits from
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen


import numpy as np
import scipy.stats as st

from rdml_graph.gaussian_process import ProbitBase
from rdml_graph.gaussian_process import std_norm_pdf, std_norm_cdf, calc_pdf_cdf_ratio


## OrdinalProbit
# This is almost directly Nick's code, for Ordinal regression.
# This is based on the paper by Chu and Ghahramani
# https://www.jmlr.org/papers/v6/chu05a.html
# Ordinal regression is for categorical scales (like a likert scale)
class OrdinalProbit(ProbitBase):
    type = 'categorical'
    y_type = 'discrete'


    ## Constructor
    # @param sigma - the sigma on the likelihood function
    # @param b - the bias parameter
    # @param n_ordinals - the number of ordinal categories.
    # @param eps - the epsilon on #TODO
    def __init__(self, sigma=1.0, b=1.0, n_ordinals=5, eps=1.0e-10):
        self.n_ordinals=n_ordinals
        self.set_hyper([sigma, b])
        self.eps = eps
        self.y_list = np.atleast_2d(np.arange(1, self.n_ordinals+1, 1, dtype='int')).T

    ## set_hyper
    # Sets the hyperparameters for the probit
    # @param hyper - an array or list with [sigma, b]
    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])
        self.set_b(hyper[1])

    ## set_b
    # Sets the bias hyperparameter
    # @param b - should be a scalar or vector of breakpoints
    def set_b(self, b):
        if not hasattr(b, "__len__"):
            b = abs(b)
            self.b = np.hstack(([-np.Inf],np.linspace(-b, b, self.n_ordinals-1), [np.Inf]))
        elif len(b) == self.n_ordinals+1:
            self.b = b
        elif len(b) == self.n_ordinals-1:
            self.b = np.hstack(([-np.Inf], b, [np.Inf]))
        else:
            raise ValueError('Specified b should be a scalar or vector of breakpoints')

    ## set_sigma
    # sets the sigma hyperparamter and calculates the inverse of sigma and inverse squared
    # @param sigma - the sigma value for the probit function.
    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isigma = 1.0/self.sigma
        self._ivar = self._isigma**2

    ## print_hyperparameters
    # prints the hyperparameter of the probit
    def print_hyperparameters(self):
        print("Ordinal probit, {0} ordered categories.".format(self.n_ordinals))
        print("Sigma: {0:0.2f}, b: ".format(self.sigma))
        print(self.b)

    ## z_k
    # returns the probit for the given probit class (the difference in the
    # latent space)
    # @param y - the labels for the given probit (rating, u) (must be a numpy array)
    # @param F - the input estimates in the latent space
    #
    # @return the vector of z_k values
    def z_k(self, y, F):
        if isinstance(y, np.ndarray):
            f = F[y[:,1]]
            y = y[:,0]
        else:
            f = F
        return self._isigma*(self.b[y] - f)

    def norm_pdf(self, y, F):
        if isinstance(y, np.ndarray):
            f = F[y[:,1]]
            y = y[:,0]
        else:
            f = F

        if isinstance(y, np.ndarray):
            f = f*np.ones(y.shape, dtype='float')       # This ensures the number of f values matches the number of y
            out = np.zeros(y.shape, dtype='float')
            for i in range(out.shape[0]):
                if y[i] != 0 and y[i] != self.n_ordinals:  # b0 = -Inf -> N(-Inf) = 0
                    z = self._isigma*(self.b[y[i]] - f[i])
                    out[i] = std_norm_pdf(z)
        else:
            out = 0
            if y!=0 and y != self.n_ordinals:
                z = self._isigma*(self.b[y] - f)
                out = std_norm_pdf(z)
        return out

    def norm_cdf(self, y, F, var_x=0.0):
        if isinstance(y, np.ndarray):
            f = F[y[:,1]]
            y = y[:,0]
        else:
            f = F
        ivar = self._isigma + var_x
        f = f*np.ones(y.shape, dtype='float')
        out = np.zeros(y.shape, dtype='float')
        for i in range(out.shape[0]):
            if y[i] == self.n_ordinals:
                out[i] = 1.0
            elif y[i] != 0:
                z = ivar*(self.b[y[i]] - f[i])
                out[i] = std_norm_cdf(z)
        return out

    ## derivatives
    # Calculates the derivatives of the probit with the given input data
    # @param y - the given set of labels for the probit (rating, u_k)
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of P(y|x,theta) with respect to F
    #       py - P(y|x,theta) for the given probit
    def derivatives(self, y, F):
        f = F[y[:,1]]
        y_orig = y
        y = y[:,0]
        l = self.likelihood(y_orig, F)
        py = np.log(l)

        # First derivative - Chu and Gharamani
        # Having issues with derivative (likelihood denominator drops to 0)
        dpy_df = np.zeros(l.shape, dtype='float')
        d2py_df2 = np.zeros(l.shape, dtype='float')
        for i in range(l.shape[0]):
            if l[i] < self.eps:
                # l2 = self.likelihood(y[i], f[i]+self.delta_f)
                # l0 = self.likelihood(y[i], f[i]-self.delta_f)
                # dpy_df[i] = -(l2-l[i])/self.delta_f/l[i]      # (ln(f))' = f'/f
                # d2py_df2[i] = (l2 - 2*l[i] + l0)/self.delta_f**2/dpy_df[i]/l[i]

                if y[i] == 1:
                    dpy_df[i] = self._isigma*self.z_k(y[i], f[i])
                    d2py_df2[i] = -self._ivar
                elif y[i] == self.n_ordinals:
                    dpy_df[i] = self._isigma*self.z_k(y[i]-1, f[i])
                    d2py_df2[i] = -self._ivar
                else:
                    z1 = self.z_k(y[i], f[i])
                    z2 = self.z_k(y[i]-1, f[i])
                    ep = np.exp(-0.5*(z1**2 - z2**2))
                    dpy_df[i] = self._isigma*(z1*ep-z2)/(ep - 1.0)
                    d2py_df2[i] = -(self._ivar*(1.0 - (z1**2 *ep - z2**2)/(ep - 1.0)) + dpy_df[i]**2)
            else:
                dpy_df[i] = -self._isigma*(self.norm_pdf(y[i], f[i]) - self.norm_pdf(y[i]-1, f[i])) / l[i]
                d2py_df2[i] = -(dpy_df[i]**2 + self._ivar*(self.norm_pdf(y[i], f[i]) - self.norm_pdf(y[i]-1, f[i])) / l[i])

        W = np.diagflat(-d2py_df2)
        return W, dpy_df, py


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, F):
        y_modified = np.copy(y)
        y_modified[:,0] -= 1
        return self.norm_cdf(y, F) - self.norm_cdf(y_modified, F)


    ## posterior_likelihood
    # TODO - not quite sure what this does.
    def posterior_likelihood(self, fhat, varhat, uvi):
        raise NotImplementedError('posterior_likelihood not implemented')

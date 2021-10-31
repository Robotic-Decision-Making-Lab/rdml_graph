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

# AbsBoundProbit.py
# Written Ian Rankin - October 2021
# Modified code by Nicholas Lawerence from here
# https://github.com/osurdml/GPtest/tree/feature/wine_statruns/gp_tools
# Used with permission.
#
# A set of Gaussian Process Probits from
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen


import numpy as np
import scipy.stats as st

from rdml_graph.gaussian_process import ProbitBase
from rdml_graph.gaussian_process import std_norm_pdf, std_norm_cdf

from scipy.special import digamma, polygamma
from scipy.stats import norm, beta

## AbsBoundProbit
# This is almost directly Nick's code, for absolute bounded inputs.
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen
#
#
class AbsBoundProbit(ProbitBase):
    type = 'bounded continuous'
    y_type = 'bounded'


    ## Constructor
    # @param sigma - the slope of the probit, basically scales how far away from
    #               0 the latent has to be to to move away from 0.5 output. Sigma should
    #               basically relate to the range of the latent function
    # @param v - the precision, kind of related to inverse of noise, high v is sharp distributions
    def __init__(self, sigma=1.0, v=10.0):
        self.set_hyper([sigma, v])
        self.log2pi = np.log(2.0*np.pi)


    ## set_hyper
    # Sets the hyperparameters for the probit
    # @param hyper - a list of hyperparameters [sigma, v]
    #               sigma, the slope of the probit
    #               v, precision, related to inverse of noise
    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])
        self.set_v(hyper[1])

    ## set_sigma
    # Sets the sigma on the absolute bounded probit.
    # Also calculates inverse to the sigma for fast calculation.
    # @param sigma - the slope of the probit, basically scales how far away from
    #               0 the latent has to be to to move away from 0.5 output. Sigma should
    #               basically relate to the range of the latent function
    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2

    ## sets the v variable.
    # @param v - the precision, kind of related to inverse of noise, high v is sharp distributions
    def set_v(self, v):
        self.v = v

    ## print_hyperparameters
    # prints the hyperparameter of the probit
    def print_hyperparameters(self):
        print("Beta distribution, probit mean link.")
        print("Sigma: {0:0.2f}, v: {1:0.2f}".format(self.sigma, self.v))


    ## mean_link
    # The mean link function for the probit function.
    # Defined as equation 11 in Section 3.2.1
    # @param F - the predicted locations
    def mean_link(self, F):
        ml = np.clip(std_norm_cdf(F*self._isqrt2sig), 1e-12, 1.0-1e-12)
        return ml

    ## get_alpha_beta
    # the alpha and beta function for the mean function.
    # Equation 10
    # @param F - the input data
    def get_alpha_beta(self, F):
        ml = self.mean_link(F)
        aa = self.v * ml
        bb = self.v - aa    # = self.v * (1-ml)
        return aa, bb

    ## derivatives
    # Calculates the derivatives of the probit with the given input data
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of P(y|x,theta) with respect to F
    #       py - P(y|x,theta) for the given probit
    def derivatives(self, y, F):
        y_sel = y[0][y[1]]

        aa, bb = self.get_alpha_beta(F)

        # Trouble with derivatives...
        dpy_df = self.v*self._isqrt2sig*std_norm_pdf(F*self._isqrt2sig) * (np.log(y_sel) - np.log(1-y_sel) - digamma(aa) + digamma(bb))

        Wdiag = - self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (
            F * self._i2var * (np.log(y_sel) - np.log(1.0-y_sel) - digamma(aa) + digamma(bb)) +
            self.v * self._isqrt2sig * std_norm_pdf(F*self._isqrt2sig) * (polygamma(1, aa) + polygamma(1, bb)) )

        W = np.diagflat(Wdiag)

        py = np.log(beta.pdf(y_sel, aa, bb))

        return -W, dpy_df, py


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit (np(float) data, np(int)index)
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, F):
        y_selected = y[0][y[1]]
        aa, bb = self.get_alpha_beta(f)
        return beta.pdf(y, aa, bb)

    ## cdf for the beta function.
    def cdf(self, y, F):
        aa, bb = self.get_alpha_beta(f)
        return beta.cdf(y, aa, bb)


    ## posterior_likelihood
    # TODO - not quite sure what this does.
    def posterior_likelihood(self, fhat, varhat, uvi):
        raise NotImplementedError('posterior_likelihood not implemented')

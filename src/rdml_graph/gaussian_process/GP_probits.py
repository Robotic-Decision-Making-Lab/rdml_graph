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

# GP_probits.py
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

import pdb




## ProbitBase
# Abstract class for a probit for the user GP
class ProbitBase:
    type = 'unknown'
    y_type = 'unknown'

    ## Constructor
    # pass the list of relational objects
    # def __init__(self, y_list):
    #     self.y_list = y_list

    ## set_hyper
    # Sets the hyperparameters for the probit
    def set_hyper(self, hyper):
        raise NotImplementedError('set_hyper not implemented')

    ## print_hyperparameters
    # prints the hyperparameter of the probit
    def print_hyperparameters(self):
        raise NotImplementedError('print_hyperparameters not implemented')

    ## z_k
    # returns the probit for the given probit class (the difference in the
    # latent space)
    # @param y - the label for the given probit
    # @param F - TODO
    def z_k(self, y, F):
        raise NotImplementedError('z_k not implemented')

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
        raise NotImplementedError('derivatives not implemented')


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, f):
        raise NotImplementedError('likelihood not implemented')

    ## log_likelihood
    # Returns the log liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return log P(y|F)
    def log_likelihood(self, y, f):
        return np.log(self.likelihood(y,f))

    ## posterior_likelihood
    # TODO - not quite sure what this does.
    def posterior_likelihood(self, fhat, varhat, uvi):
        raise NotImplementedError('posterior_likelihood not implemented')




## PreferenceProbit
# A relative discrete probit
# Partially taken from Nick's code
# this calculates the probability of preference labels from the latent space
class PreferenceProbit(ProbitBase):
    type = 'preference'
    y_type = 'discrete'

    ## constructor
    def __init__(self, sigma):
        self.set_sigma(sigma)
        self.log2pi = np.log(2.0*np.pi)

    ## set_hyper
    # Sets the hyperparameters for the probit
    # @param hyper - a sequence with [sigma]
    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])

    ## set_sigma
    # Sets the sigma value, and also sets up a couple of useful constants to go with it.
    # @param sigma - the sigma to use
    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2


    ## print_hyperparameters
    # prints the hyperparameter of the probit
    def print_hyperparameters(self):
        print("Probit relative, Gaussian noise on latent. Sigma: {0:0.2f}".format(self.sigma))

    ## z_k
    # returns the probit for the given probit class (the difference in the
    # latent space)
    # @param y - the label for the given probit (dk, u, v) (must be a numpy array)
    # @param F - the input estimates in the latent space
    #
    # @return the vector of z_k values
    def z_k(self, y, F):
        return self._isqrt2sig * y[:,0] * (F[y[:,2]] - F[y[:,1]])

    ## derv_discrete_loglike
    # Calculates the first derivative of log likelihood.
    # Appendix A.1.1.1.1
    # Assumes (f(vk), f(uk))
    #
    # xi, yk, vk, are indicies of the likelihood
    # @param y - the label for the given probit (dk, u, v) (must be a numpy array)
    # @param F - the vector of F (estimated training sample outputs)
    #
    # @return - the values for the u and v of y numpy (n,2) [[]]
    def derv_log_likelyhood(self, y, F):
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(self.z_k(y, F))
        derv_ll = y[:,0] * pdf_cdf_ratio * self._isqrt2sig

        return np.array([derv_ll, -derv_ll])

    ## derivatives
    # Calculates the derivatives of the probit with the given input data
    # @param y - the given set of labels for the probit
    #              this is given as a list of [(dk, u, v), ...]
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of log P(y|x,theta) with respect to F
    #       py - log P(y|x,theta) for the given probit
    def derivatives(self, y, F):
        py = self.log_likelihood(y, F)

        dpy_df = self.derv_log_likelyhood(y, F)

        return W, dpy_df, py


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, f):
        raise NotImplementedError('likelihood not implemented')

    ## posterior_likelihood
    # TODO - not quite sure what this does.
    def posterior_likelihood(self, fhat, varhat, uvi):
        raise NotImplementedError('posterior_likelihood not implemented')










def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return st.norm.pdf(x)
    # return np.exp(-(x**2)/2)/_sqrt_2pi


def std_norm_cdf(x):
    x = np.clip(x, -30, 100 )
    return st.norm.cdf(x)


# def norm_pdf_norm_cdf_ratio(z):
#     # Inverse Mills ratio for stability
#     out = -z
#     out[z>-30] = std_norm_pdf(z[z>-30])/std_norm_cdf(z[z>-30])
#     return out

def calc_pdf_cdf_ratio(z):
    # as zk -> -infinity then pdf_zk / cdf_zk goes to infinity
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D%2Cx-%3E-%E2%88%9E%5D
    # As zk -> infinity then pdf_zk / cdf_zk goes to 0
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D%2Cx-%3E%E2%88%9E%5D
    # this code checks for those states and inputs the appropriate values
    #

    ########### TODO vectorize
    pdf_z = std_norm_pdf(z)
    cdf_z = std_norm_cdf(z)

    pdf_cdf_z = np.where(cdf_z == 0, \
                    np.where(pdf_z < 0.5, float('inf'), 0), \
                    pdf_z / cdf_z)

    pdf_cdf_2 = np.where(np.isinf(pdf_cdf_z), 0,  pdf_cdf_z*pdf_cdf_z)

    return pdf_cdf_z, pdf_cdf_2

























#

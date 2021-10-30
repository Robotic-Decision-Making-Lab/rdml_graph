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

# PreferenceProbit.py
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
        derv_ll_pairs = y[:,0] * pdf_cdf_ratio * self._isqrt2sig
        derv_ll = np.zeros(len(F))


        derv_ll = add_up_vec(y[:,1], -derv_ll_pairs, derv_ll)
        derv_ll = add_up_vec(y[:,2], +derv_ll_pairs, derv_ll)

        # for i in range(len(y)):
        #     derv_ll[y[i,1]] -= derv_ll_pairs[i]
        #     derv_ll[y[i,2]] += derv_ll_pairs[i]

        return derv_ll

    ## calc_W
    # caclulate the W matrix
    # Calculates the second derivative of discrete log likelihood.
    # @param y - the given set of labels for the probit
    #              this is given as a list of [(dk, u, v), ...]
    #  d / (dF(xi)dF(xj)) ln(p(dk|F(u), F(v)))
    #
    # Appendix A.1.1.2
    # Assumes (f(vk), f(uk))
    # xi, yk, vk, are indicies of the likelihood
    # @param F - the vector of f (estimated training sample outputs)
    def calc_W(self, y, F):
        z = self.z_k(y, F)
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(z)

        paren_pairs = np.where(np.logical_and(z < 0, np.isinf(pdf_cdf_ratio)), 0, \
                        (z * pdf_cdf_ratio) + pdf_cdf_ratio2)
        d2_ll_pairs = -(y[:,0]*y[:,0])*paren_pairs*self._i2var

        W = np.zeros((len(F), len(F)))

        # vectorized versions of summation
        idx1 = np.array([y[:,1], y[:,1]]).T
        idx2 = np.array([y[:,1], y[:,2]]).T
        idx3 = np.array([y[:,2], y[:,1]]).T
        idx4 = np.array([y[:,2], y[:,2]]).T

        W = add_up_mat(idx1, -d2_ll_pairs, W)
        W = add_up_mat(idx2,  d2_ll_pairs, W)
        W = add_up_mat(idx3,  d2_ll_pairs, W)
        W = add_up_mat(idx4, -d2_ll_pairs, W)

        # for i in range(len(y)):
        #     W[y[i,1], y[i,1]] += -d2_ll_pairs[i]
        #     W[y[i,1], y[i,2]] +=  d2_ll_pairs[i]
        #     W[y[i,2], y[i,1]] +=  d2_ll_pairs[i]
        #     W[y[i,2], y[i,2]] += -d2_ll_pairs[i]

        return W


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
        W = self.calc_W(y, F)

        return W, dpy_df, py


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, F):
        z = self.z_k(y, F)
        return std_norm_cdf(z)

    ## posterior_likelihood
    # TODO - not quite sure what this does.
    def posterior_likelihood(self, fhat, varhat, uvi):
        raise NotImplementedError('posterior_likelihood not implemented')







try:
    import numba

    ## add_up_mat
    # add the values in v to the M matrix indexed by the indicies matrix
    # @param indicies - (n x k) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the matrix to add up
    @numba.jit
    def add_up_mat(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i,0], indicies[i,1]] += v_i

        return M

    ## add_up_vec
    # add the values in v to the M vector indexed by the indicies matrix
    # @param indicies - (n,) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the vector to add up
    @numba.jit
    def add_up_vec(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i]] += v_i

        return M
except ImportError:
    print('Failed to import numba, add_up_mat and add_up_vec will be slower')

    ## add_up_mat
    # add the values in v to the M matrix indexed by the indicies matrix
    # @param indicies - (n x k) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the matrix to add up
    def add_up_mat(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i,0], indicies[i,1]] += v_i

        return M

    ## add_up_vec
    # add the values in v to the M vector indexed by the indicies matrix
    # @param indicies - (n,) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the vector to add up
    def add_up_vec(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i]] += v_i

        return M























#

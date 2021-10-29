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
    # def z_k(self, y, F):
    #     raise NotImplementedError('z_k not implemented')

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
    def likelihood(self, y, F):
        raise NotImplementedError('likelihood not implemented')

    ## log_likelihood
    # Returns the log liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return log P(y|F)
    def log_likelihood(self, y, F):
        return np.sum(np.log(self.likelihood(y,F)))

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


## OrdinalProbit
# This is almost directly Nick's code, for Ordinal regression.
# This is based on the paper by Chu and Ghahramani
# https://www.jmlr.org/papers/v6/chu05a.html
# Ordinal regression is for categorical scales (like a likert scale)
class OrdinalProbit:
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

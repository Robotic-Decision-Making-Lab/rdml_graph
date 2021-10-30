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

# ProbitBase.py
# Written Ian Rankin - October 2021
#
# A set of different Gaussian Process Probits from
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen


import numpy as np
import scipy.stats as st




def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return st.norm.pdf(x)
    # return np.exp(-(x**2)/2)/_sqrt_2pi


def std_norm_cdf(x):
    x = np.clip(x, -30, 100 )
    return st.norm.cdf(x)



def calc_pdf_cdf_ratio(z):
    # as zk -> -infinity then pdf_zk / cdf_zk goes to infinity
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D%2Cx-%3E-%E2%88%9E%5D
    # As zk -> infinity then pdf_zk / cdf_zk goes to 0
    # https://www.wolframalpha.com/input/?i2d=true&i=Limit%5BDivide%5BPower%5B%5C%2840%29Exp%5B-Divide%5BPower%5Bx%2C2%5D%2C2%5D%5D%5C%2841%29%2C2%5D%2CPower%5Berfc%5C%2840%29-Divide%5Bx%2CSqrt%5B2%5D%5D%5C%2841%29%2C2%5D%5D%2Cx-%3E%E2%88%9E%5D
    # this code checks for those states and inputs the appropriate values
    pdf_z = std_norm_pdf(z)
    cdf_z = std_norm_cdf(z)

    pdf_cdf_z = np.where(cdf_z == 0, \
                    np.where(pdf_z < 0.5, float('inf'), 0), \
                    pdf_z / cdf_z)

    pdf_cdf_2 = np.where(np.isinf(pdf_cdf_z), 0,  pdf_cdf_z*pdf_cdf_z)

    return pdf_cdf_z, pdf_cdf_2



## ProbitBase
# Abstract class for a probit for the user GP
class ProbitBase:
    type = 'unknown'
    y_type = 'unknown'


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

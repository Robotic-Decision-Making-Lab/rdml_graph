# Copyright 2023 Ian Rankin
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

# PreferenceLinear.py
# Written Ian Rankin - November 2023
#
# A linear latent function to learn the given preferences.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

from rdml_graph.gaussian_process import PreferenceProbit, ProbitBase
from rdml_graph.gaussian_process import k_fold_half, get_dk
from rdml_graph.gaussian_process import PreferenceModel

import pdb

class PreferenceLinear(PreferenceModel):
    ## init function
    # @param pareto_pairs - [opt] sets whether to assume pareto optimal user preferences.
    # @param other_probits - [opt] sets additional types of probits to add.
    def __init__(self, pareto_pairs=False, other_probits={}):
        super(PreferenceLinear, self).__init__(pareto_pairs, other_probits)


        self.probits = [PreferenceProbit(sigma = 1.0)]
        
        self.lambda_newton = 0.3
        for key in other_probits:
            if not isinstance(other_probits[key], ProbitBase):
                raise TypeError("PreferenceLinear pased a probit that is not a probit: " + str(other_probits[key]))

            self.probits.append(other_probits[key])        



    ## P_w
    # Probability of w given the training data
    def log_P_w(self, w):
        log_p_w = 0.0
        for j, probit in enumerate(self.probits):
            if self.y_train[j] is not None:
                p_w_local = probit.log_likelihood(self.y_train[j], F)

                log_p_w += p_w_local

        return log_p_w
        

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
    def derivatives(self, x, y, w):
        F = (x @ w[:,np.newaxis])[:,0]

        W = np.zeros((len(F), len(F)))
        grad_ll = np.zeros(len(F))
        log_likelihood = 0
        for j, probit in enumerate(self.probits):
            if self.y_train[j] is not None:
                W_local, dpy_df_local, py_local = probit.derivatives(y[j], F)

                W += W_local
                grad_ll += dpy_df_local
                log_likelihood += py_local


        # need to multiply by derivative of dl/df * df/dw
        grad_ll = (grad_ll[np.newaxis,:] @ x)[0]
        W = x.T @ W @ x

        return W, grad_ll, log_likelihood

    ## optimize
    # Runs the optimization step required by the user preference GP.
    # @param optimize_hyperparameter - [opt] sets whether to optimize the hyperparameters
    def optimize(self, optimize_hyperparameter=False):
        if len(self.X_train.shape) > 1:
            self.w = np.random.random(self.X_train.shape[1])
        else:
            print('Only 1 reward parameter... linear model practically does not make sense')
            self.w = np.random.random(1)


        # just do gradient decent.
        W, dpy_dw, py = self.derivatives(self.X_train, self.y_train, self.w)

        pdb.set_trace()

        self.optimized = True





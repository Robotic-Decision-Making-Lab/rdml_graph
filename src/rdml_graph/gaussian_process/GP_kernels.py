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

## RBF_kern
# Radial basis function for two points.
# This is a single input sample (v (k) and u (k))
# @param u - a single input sample
# @param v - a second input sample of the same dimension
# @param cov_data - the input covariance data {'rbf_sigma', 'rbf_l'}
#               rbf_sigma, the sigma for the rbf kernel
#               rbf_l, the lengthscale for the rbf_kernel
#
# @return - a scalar value for the covariance between the two samples.
def RBF_kern(u, v, cov_data):
    sigma = cov_data['rbf_sigma']
    l = cov_data['rbf_l']

    top = (u - v)
    top = np.sum(top * top)

    return sigma**2 * np.exp(-top / (2 * l * l))


## periodic_kern
# A periodic kernel for gaussian processes
# @param u - a single input sample
# @param v - a second input sample of the same dimension
# @param cov_data - the input covariance data {'rbf_sigma', 'rbf_l'}
#               periodic_sigma, the sigma for the periodic kernel
#               periodic_l, the lengthscale for the periodic kernel
#               periodic_p, the periodicity of the periodic kernel
#
# @return - a scalar value for the covariance between the two samples.
def periodic_kern(u, v, cov_data):
    sigma = cov_data['periodic_sigma']
    l = cov_data['periodic_l']
    p = cov_data['periodic_p']

    sin_tmp = np.sin(np.pi*np.abs(u - v) / p)
    exp_int = - 2 * sin_tmp * sin_tmp / (l*l)

    return sigma*sigma * np.exp(exp_int)

## linear_kern
# A periodic kernel for gaussian processes
# @param u - a single input sample
# @param v - a second input sample of the same dimension
# @param cov_data - the input covariance data {'rbf_sigma', 'rbf_l'}
#               linear_sigma, the sigma for the linear kernel
#               linear_sigma_b, the additive sigma for the linear kernel
#               linear_offset, the offcet value of the periodic kernel
#
# @return - a scalar value for the covariance between the two samples.
def linear_kern(u, v, cov_data):
    sigma = cov_data['linear_sigma']
    sigma_b = cov_data['linear_sigma_b']
    c = cov_data['linear_offset']

    return (sigma_b**2) + ((sigma**2) * (u-c) * (v-c))

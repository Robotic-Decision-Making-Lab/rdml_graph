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

# SimpleGP.py
# Written Ian Rankin - September 2021
#
# An example usage of a simple Gaussian process

import numpy as np
import matplotlib.pyplot as plt

import rdml_graph as gr

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


if __name__ == '__main__':
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = gr.generate_fake_pairs(X_train, f_sin, 0) + \
            gr.generate_fake_pairs(X_train, f_sin, 1) + \
            gr.generate_fake_pairs(X_train, f_sin, 2) + \
            gr.generate_fake_pairs(X_train, f_sin, 3) + \
            gr.generate_fake_pairs(X_train, f_sin, 4)


    gp = gr.PreferenceGP(gr.RBF_kern, {'rbf_sigma': 1, 'rbf_l': 0.5})
    #gp = gr.PreferenceGP(gr.periodic_kern, {'periodic_sigma': 1, 'periodic_l': 1, 'periodic_p': 20})
    #gp = gr.PreferenceGP(gr.linear_kern, {'linear_sigma': 5, 'linear_sigma_b': 5, 'linear_offset': 0.2})
    # gp = gr.PreferenceGP(
    #             lambda u,v, d : gr.RBF_kern(u,v,d) + gr.periodic_kern(u,v,d) + gr.linear_kern(u,v,d),
    #             {'rbf_sigma': 1, 'rbf_l': 1,
    #             'periodic_sigma': 1, 'periodic_l': 1, 'periodic_p': 10,
    #             'linear_sigma': 3, 'linear_sigma_b': 1, 'linear_offset': 0.3})
    # gp = gr.PreferenceGP(
    #             lambda u,v, d : gr.RBF_kern(u,v,d) + gr.linear_kern(u,v,d),
    #             {'rbf_sigma': 1, 'rbf_l': 1,
    #             'linear_sigma': 3, 'linear_sigma_b': 1, 'linear_offset': 0.3})

    gp.sigma_L = 1.0

    gp.add(X_train, pairs)



    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    plt.plot(X, mu)
    sigma_to_plot = 1

    plt.gca().fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    plt.plot(X, f_sin(X))
    plt.scatter(X_train, f_sin(X_train))
    #plt.scatter(X_train, F)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

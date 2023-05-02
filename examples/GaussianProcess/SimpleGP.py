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


if __name__ == '__main__':
    X_train = np.array([0,1,2,3,6,7])
    X = np.arange(-3, 12, 0.1)
    y_train = np.array([1, 0.5,0, -1, 1, 2])
    #y_train = np.array([0.2, 0.3, 0.4, 0.52, 0.7, 0.76])

    #training_sigma = 0
    training_sigma=np.array([1, 0.5, 0.1, 0.1, 0.2, 0])

    ##gp = gr.GP(gr.RBF_kern, {'rbf_sigma': 1, 'rbf_l': 1})
    ##gp = gr.GP(gr.periodic_kern, {'periodic_sigma': 1, 'periodic_l': 1, 'periodic_p': 20})
    ##gp = gr.GP(gr.linear_kern, {'linear_sigma': 5, 'linear_sigma_b': 5, 'linear_offset': 0.2})
    gp = gr.GP(gr.RBF_kern(1,1)+gr.periodic_kern(1,1,10)+gr.linear_kern(3,1,0.3))
    #gp = gr.GP(gr.RBF_kern(1,1))
    #gp = gr.GP(gr.linear_kern(3,5,3))
    #gp = gr.GP(gr.periodic_kern(1, 1, 20))
    gp.add(X_train, y_train, training_sigma=training_sigma)

    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    plt.plot(X, mu)
    sigma_to_plot = 1

    plt.gca().fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    plt.scatter(X_train, y_train)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

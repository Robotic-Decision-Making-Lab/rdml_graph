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

# OrdinalGP.py
# Written Ian Rankin - September 2021
#
# An example usage of a GP with an ordinal scale


import numpy as np
import matplotlib.pyplot as plt

import rdml_graph as gr




if __name__ == '__main__':
    X_train = np.array([0,1,2,3,4.2,6,7])
    ratings = np.array([5,5,2,1,2  ,3,3])


    gp = gr.PreferenceGP(gr.RBF_kern(0.5, 0.7), \
            other_probits={'ordinal': gr.OrdinalProbit(2.0,1.0, n_ordinals=5)})
    #gp = gr.PreferenceGP(gr.periodic_kern(1.2,0.3,5))
    #gp = gr.PreferenceGP(gr.linear_kern(0.2, 0.2, 0.2))
    #gp = gr.PreferenceGP(gr.RBF_kern(0.2,1)+gr.periodic_kern(1,0.2,0)+gr.linear_kern(0.2,0.1,0.3))
    #gp = gr.PreferenceGP(gr.RBF_kern(0.1,1)*gr.linear_kern(0.3,0.2,0.3))

    gp.add(X_train, ratings, type='ordinal')

    #gp.optimize(optimize_hyperparameter=True)
    #print('gp.calc_ll()')
    #print(gp.calc_ll())


    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    plt.scatter(X_train, ratings)
    sigma_to_plot = 1

    plt.plot(X, mu)
    plt.scatter(X_train, gp.F)
    plt.gca().fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    #plt.scatter(X_train, F)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Predicted function', 'Given ratings', 'Predicted ratings'])
    plt.show()

# Copyright 2020 Ian Rankin
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
#
# TestInfoField.py
# Written Ian Rankin January 2021
#
#

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt



info_field = gr.random_field2d((50,50))

multi_field = gr.random_multi_field2d((50,50), 3, num_gauss=np.array([5,7,9]))

mat1 = plt.matshow(info_field)
cbar = plt.colorbar(mat1)
#plt.show()
plt.figure()

mat = plt.imshow(multi_field, vmin=np.min(multi_field), vmax=np.max(multi_field))
#cbar = plt.colorbar(mat)

for i in range(multi_field.shape[2]):
    plt.figure()
    mat_i = plt.matshow(multi_field[:,:,i])
    cbar = plt.colorbar(mat_i)

paths = [np.array([[6,22], [16, 11], [43, 34]]), \
         np.array([[6,22], [18, 47]])]

for i in range(5):
    paths.append(np.random.random((4,2))*49)


gr.plot_multi(multi_field, paths, cmap='Greys')

plt.show()

# TestGPKernels.py
# Written Ian Rankin October 2021
#
#

import numpy as np
import rdml_graph as gr
import pdb


rbf = gr.RBF_kern(1, 1)
kern2 = gr.periodic_kern(1,1, 3)
kern3 = gr.linear_kern(1,1,1)

print(rbf)
print(kern2)
print(kern3)
print(rbf(1,2))
print(kern2(1,2))
print(kern3(1,2))

combined = rbf + (kern2 * kern3)
print(combined)
print(combined(1,2))

print(combined.get_param())
print(combined.gradient(1,2))
#combined.set_param([2,3,5,4,7, 1,1,2])
#print(combined.get_param())

#x = np.array([0,1,4,5,6,7])

#print(combined.cov(x,x))

rbf_test = gr.RBF_kern(1, 1)

X = np.array([0,1,2,3,4.2,6,7])


c = rbf_test.cov(X,X)
print(c)

print(rbf_test(0,1))
pdb.set_trace()

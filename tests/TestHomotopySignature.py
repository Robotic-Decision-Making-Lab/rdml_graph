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

# TestHomotopySignature.py
# Written Ian Rankin - October 2020
#
# A test function for the homotopy signature class.

import rdml_graph as gr
import numpy as np

n = gr.GeometricNode(0,np.array([0,1]))
n1 = gr.GeometricNode(1, np.array([1,2]))
n2 = gr.GeometricNode(2, np.array([3,4]))
n3 = gr.GeometricNode(3, np.array([4.4,-3]))
n4 = gr.GeometricNode(4, np.array([-2.2,-10]))
n5 = gr.GeometricNode(5, np.array([4.5,3.2]))


sign = gr.HomotopySignature([])
sign.cross(1, 1)
print(sign)
sign.cross(2,1)
print(sign)
sign.cross(2,-1)
print(sign)
sign.cross(2,-1)
print(sign)
sign.cross(2,1)
print(sign)
sign.cross(2,1)
print(sign)
sign.cross(2,1)
print(sign)
print('\n\n')
sign2 = gr.HomotopySignature([-2, -2, 1])

sign3 = sign + sign2
print(sign)
print(sign2)
print(sign3)

print(sign - sign2)

features = np.array([[0,0], [2,2]])

e = gr.HEdge(n,n1, gr.HomotopySignature())
print(e)
e = gr.HEdge(n,n1, gr.HomotopySignature(), features=features)
print(e)
e2 = gr.HEdge(n1,n2, gr.HomotopySignature(), features=features)
print(e2)
e3 = gr.HEdge(n,n2, gr.HomotopySignature(), features=features)
print(e3)
e4 = gr.HEdge(n2,n, gr.HomotopySignature(), features=features)
print(e4)

























#

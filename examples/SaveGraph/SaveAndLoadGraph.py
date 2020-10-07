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
# SaveAndLoadGraph.py
# Written Ian Rankin - March 2020
#
# An example set of code to show loading and saving a graph structure using pickle

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt

import pickle

features = np.array([[0,0]])

n = gr.GeometricNode(0,np.array([-3,-3]))
n1 = gr.GeometricNode(1, np.array([-3.3,3]))
n2 = gr.GeometricNode(2, np.array([3.4,-2]))
n3 = gr.GeometricNode(3, np.array([4.4,-3]))
n4 = gr.GeometricNode(4, np.array([-2.2,-10]))
n5 = gr.GeometricNode(5, np.array([4.5,3.2]))

G = [n,n1,n2,n3,n4,n5]

n.addEdge(gr.HEdge(n,n1, gr.HomologySignature(), features=features))
n1.addEdge(gr.HEdge(n1,n, gr.HomologySignature(), features=features))

n.addEdge(gr.HEdge(n,n2, gr.HomologySignature(), features=features))
n2.addEdge(gr.HEdge(n2,n, gr.HomologySignature(), features=features))

n.addEdge(gr.HEdge(n,n3, gr.HomologySignature(), features=features))
n3.addEdge(gr.HEdge(n3,n, gr.HomologySignature(), features=features))

n2.addEdge(gr.HEdge(n2,n4, gr.HomologySignature(), features=features))
n4.addEdge(gr.HEdge(n4,n2, gr.HomologySignature(), features=features))

n1.addEdge(gr.HEdge(n1,n5, gr.HomologySignature(), features=features))
n5.addEdge(gr.HEdge(n5,n1, gr.HomologySignature(), features=featuress))

n3.addEdge(gr.HEdge(n3,n5, gr.HomologySignature(), features=features))
n5.addEdge(gr.HEdge(n5,n3, gr.HomologySignature(), features=features))

n.addEdge(gr.HEdge(n,n5, gr.HomologySignature(), features=features))
n5.addEdge(gr.HEdge(n5,n, gr.HomologySignature(), features=features))

gr.plot2DGeoGraph(G, 'blue')
plt.title('Original')
plt.show(False)


pickle.dump(G, open("sample.g", "wb"))

loaded = pickle.load( open("sample.g", "rb"))

plt.figure()
gr.plot2DGeoGraph(loaded, 'red')
plt.title('Loaded graph')
plt.show()

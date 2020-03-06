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

n.addEdge(gr.HomotopyEdge(n,n1, features=features))
n1.addEdge(gr.HomotopyEdge(n1,n, features=features))

n.addEdge(gr.HomotopyEdge(n,n2, features=features))
n2.addEdge(gr.HomotopyEdge(n2,n, features=features))

n.addEdge(gr.HomotopyEdge(n,n3, features=features))
n3.addEdge(gr.HomotopyEdge(n3,n, features=features))

n2.addEdge(gr.HomotopyEdge(n2,n4, features=features))
n4.addEdge(gr.HomotopyEdge(n4,n2, features=features))

n1.addEdge(gr.HomotopyEdge(n1,n5, features=features))
n5.addEdge(gr.HomotopyEdge(n5,n1, features=features))

n3.addEdge(gr.HomotopyEdge(n3,n5, features=features))
n5.addEdge(gr.HomotopyEdge(n5,n3, features=features))

n.addEdge(gr.HomotopyEdge(n,n5, features=features))
n5.addEdge(gr.HomotopyEdge(n5,n, features=features))

gr.plot2DGeoGraph(G, 'blue')
plt.title('Original')
plt.show(False)


pickle.dump(G, open("sample.g", "wb"))

loaded = pickle.load( open("sample.g", "rb"))

plt.figure()
gr.plot2DGeoGraph(loaded, 'red')
plt.title('Loaded graph')
plt.show()

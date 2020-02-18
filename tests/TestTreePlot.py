# TestTreePlot.py
# Written Ian Rankin - February 2020
#
# A test script for plotting a tree structure.
#

import rdml_graph as gr




n = gr.Node(0)
n1 = gr.Node(1)
n2 = gr.Node(2)
n3 = gr.Node(3)
n4 = gr.Node(4)
n5 = gr.Node(5)


n.addEdge(gr.Edge(n,n1))
n.addEdge(gr.Edge(n,n2))
n.addEdge(gr.Edge(n,n3))
n2.addEdge(gr.Edge(n2,n4))
n1.addEdge(gr.Edge(n1,n5))

gr.plotTree(n, show_labels=True)

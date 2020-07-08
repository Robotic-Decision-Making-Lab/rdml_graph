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

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
#
# DecisionNodes.py
# Written Ian Rankin - March 2021

from rdml_graph.core import TreeNode, Edge
import random

import pdb



# While it currently does nothing, this leaves me with the option to add something
# to all decision node types. (Potentially a function for explanation etc.)
class DecisionNode(TreeNode):
    # @param id - the id of the decision node (needs to be a unique integer)
    # @param parent - the parent of the decision node
    def __init__(self, id, parent):
        super(DecisionNode, self).__init__(id, parent)
        self.idx = None # Should be overriden by non-abstract classes
        self.samples=samples

    # set_node
    # This function sets the connector nodes for
    def set_node(self, edge_num, node):
        self.e[edge_num].c = node

    # traverses the tree to get to the leaf node.
    def traverse(self, input):
        next = self.get_next(input)
        if isinstance(next, DecisionNode):
            return next.traverse(input)
        else:
            return next

    # @override
    # @param X - the input data
    # @param with_label - set true if the input data includes
    def separate(self, X, with_label=False):
        splits = [[] for i in range(len(self.e))]

        for x in X:
            if with_label:
                x_data = x[0]
            else:
                x_data = x

            for i, edge in enumerate(self.e):
                if x_data[self.idx] in edge:
                    splits[i].append(x)
                    break # can stop early.
        # end for loop over X
        return splits

    # This function returns the next node in the tree given the input (or a leaf object)
    # this is not super effcient, but fairly elegant.
    # @param input - given the current input return the output
    #
    # @return - the next node or an object for a leaf node.
    def get_next(self, input):
        for edge in self.e:
            if input[self.idx] in edge:
                return edge.c
        # category not listed
        raise ValueError('Given input ' + str(input) +' decision node: ' + str(self.e))

    # A set of code to get visualization code for a tree.
    # This uses the graphviz python library to generate a Digraph object for tree.
    # @oaram labels - boolean for if the tree should inlcude lables.
    # @param t - a Digraph object to start with (leave if creating a new viz)
    #
    # @return - Digraph object ( t.view() ) called after will show the tree.
    def get_viz(self, labels=False, t=None):
        if t is None:
            from graphviz import Digraph
            t = Digraph('T')

        if labels:
            label = self.get_plot_label()
            t.node(str(self.id), label)
        else:
            t.node(str(self.id), '')

        if self.parent is not None:
            t.edge(str(self.parent.id), str(self.id))

        # recursivly call sub calls
        for e in self.e:
            if isinstance(e.c, gr.TreeNode):
                t = e.c.get_viz(labels, t)
            else:
                class_label='label_'+str(random.randint(1000000, 100000000))
                t.node(class_label, str(e.c))
                t.edge(str(self.id), class_label)
        return t


class DecisionEdge(Edge):
    def __init__(self, parent, child):
        super(DecisionEdge, self).__init__(parent, child)

    def __contains__(self, item):
        return NotImplementedError('DecisionEdge function not implemented')

class CategoryEdge(DecisionEdge):
    def __init__(self, parent, child, categories):
        super(CategoryEdge, self).__init__(parent, child)
        if isinstance(categories, list):
            self.categories = set(categories)
        else:
            self.categories = set([categories])

    def __contains__(self, item):
        return item in self.categories

class FloatEdge(DecisionEdge):
    def __init__(self, parent, child, value, larger=True):
        super(FloatEdge, self).__init__(parent, child)
        self.value = value
        self.larger = larger


    def __contains__(self, item):
        return not (bool(item > self.value) ^ bool(self.larger))


# A bi-decision decision node
# All values greater than value are in the second edge.
class FloatDecision(DecisionNode):
    def __init__(self, id, parent, idx, value):
        super(DecisionNode, self).__init__(id, parent)
        self.idx = idx

        self.e = [  FloatEdge(self, None, value, False), \
                    FloatEdge(self, None, value, True)]


    def set_upper(self, node):
        self.e[1].c = node
    def set_lower(self, node):
        self.e[0].c = node

    def get_plot_label(self, data=None):
        s = 'idx: ' + str(self.idx) + ' > ' + str(self.e[0].value)
        return s


class CategoryDecision(DecisionNode):
    # @param id - the id of the decision node (needs to be a unique integer)
    # @param parent - the parent of the decision node
    # @param idx - the index in the input values of the categories
    # @param categories - a list of lists specifying the desired categories
    def __init__(self, id, parent, idx, categories):
        super(DecisionNode, self).__init__(id, parent)
        self.idx = idx

        self.e = [CategoryEdge(self, None, cat) for cat in categories]

    def get_plot_label(self, data=None):
        s = 'idx: ' + str(self.idx) + ' cat0: ' + str(self.e[0].categories) \
                + ' cat1: ' + str(self.e[1].categories)
        return s

    def __str__(self):
        o = 'CategoryDecision: ' + str(id) + ' idx: ' + str(self.idx) + ' categoies:'

        for edge in self.e:
            o += ' ' + str(edge.categories)

        return o



# split_list
# splits a list into sub lists with each element going to the list specified in
# idxs
# @param lis - the input list of objects to split into smaller list
# @param idxs - the input list of indcies each object in lis should be split into
# @param largest_idx - an optional index to setup the list to run faster if the
#                   the number of sub lists is known.
def split_list(lis, idxs, largest_idx=None):
    if largest_idx is None:
        splits = {}
        raise Exception()
        for i in range(len(idxs)):
            splits[idxs[i]].append(lis[i])
        return splits
    else:
        counts = [0] * largest_idx

        for i in range(len(idxs)):
            counts[idxs[i]] += 1
        splits = [[None] * counts[i] for i in range(largest_idx)]

        counts = [0] * largest_idx

        for i in range(len(idxs)):
            j = idxs[i]
            splits[j][counts[j]] = lis[i]
            counts[j] += 1
        return splits



if __name__ == '__main__':
    # create fake tree for quick test.
    root = FloatDecision(0, None, 0, 0.25)
    c1 = CategoryDecision(1, root, 1, [['apples'], ['oranges', 'peaches']])
    c1.set_node(0, 1)
    c1.set_node(1, 2)
    root.set_upper(c1)
    root.set_lower(0)

    X = [[0.5, 'apples'], [0.75, 'oranges'], [0, 'literally does not matter'], [1.2, 'peaches']]

    out_y = [root.traverse(x) for x in X]

    print(X)
    print(out_y)

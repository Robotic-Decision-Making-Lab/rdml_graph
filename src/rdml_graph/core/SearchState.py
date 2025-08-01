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
## @package SearchState.py
# Written Ian Rankin - January 2020
#
# A basic state for search functions
# Should be extended for new functionality

from rdml_graph.core import State
from rdml_graph.core import TreeNode
from rdml_graph.core import Edge

import pdb


class SearchState(TreeNode):
    __search_state_id_num_global__ = 0
    ## Constructor
    # @param state - should be of class State
    # @param rCost - the real cost to the state
    # @param hCost - the huerestic cost to the goal from this state
    # @param parent - the parent SearchState
    # @param id - the unique id of the object.
    def __init__(self, state, rCost=0, hCost=0, parent=None, id=None, parent_e_id=None):
        super(SearchState, self).__init__(SearchState.__search_state_id_num_global__, parent)

        # init code for id handling.
        if id is not None:
            SearchState.__search_state_id_num_global__ = id
            self.id = id
        SearchState.__search_state_id_num_global__ += 1

        #if not isinstance(state, State):
        #    raise TypeError("Search state given argument state not of type 'State', instead is type: " + str(type(state)))
        self.rCost = rCost # real cost
        self.hCost = hCost # estimated cost
        self.state = state # The actual state
        self.parent_e_id = parent_e_id  # The id of the edge that this state is a child of
        # self.parent = parent # Pointer to parent node of state for finding full path
        self.invertCmp = False # allows inverting invert comparison (say for max heap)

    ## @var invertCmp
    # sets whether the comparision should be inverted. This allows for flipping things
    # like a max heap etc.

    ## successor
    # Gets the successors of the current search state.
    # It calls the states successor function to update the search states
    #
    # @return - a list of SearchState of successors of the current search state.
    def successor(self):
        # consider how to handle the estimated hCost
        succ = self.state.successor()
        states = [SearchState(s[0], rCost=self.rCost+s[1], parent=self, parent_e_id=i)  \
                    for i, s in enumerate(succ)]
        self.e = [Edge(self, s, suc[1]) for s, suc in zip(states, succ)]
        self.calcSucc = True
        return states


    ## cost
    # returns the combination of real cost and estimated additional cost
    def cost(self):
        return self.rCost + self.hCost

    ## getRevPath
    # This function works its way to up the search tree to the root node
    # to return the list of all states in path.
    def getRevPath(self):
        if self.parent is None:
            return [self.state]

        return [self.state] + self.parent.getRevPath()


    ## getPath
    # This function works its way to up the search tree to the root node
    # to return the list of all states in path.
    def getPath(self, keepEdges=False, keepNodes=True):
        if self.parent is None:
            if keepNodes:
                return [self.state]
            else:
                return []

        if keepEdges and keepNodes:
            return self.parent.getPath(keepEdges=True) + \
                    [self.parent.state.e[self.parent_e_id], self.state]
        elif keepEdges:
            return self.parent.getPath(keepEdges=True, keepNodes=False) + \
                    [self.parent.state.e[self.parent_e_id]]
        elif keepNodes:
            return self.parent.getPath() + [self.state]
        else:
            raise ValueError("Cannot keep neither edges nor nodes in path, please select one or both of them.")

    ################################ Operator overloads

    ## < operator overload
    # Redefine less than operator for heapq
    # Comparison between costs performed
    def __lt__(self, other):
        if self.invertCmp:
            return self.cost() > other.cost()
        else:
            return self.cost() < other.cost()

    ## > operator overload
    # Redefine greater than operator for heapq
    # Comparison between costs performed
    def __gt__(self, other):
        if not self.invertCmp:
            return self.cost() > other.cost()
        else:
            return self.cost() < other.cost()

    ## == operator overload
    def __eq__(self, other):
        if not isinstance(other, SearchState):
            return False
        return self.cost() == other.cost()

    ## a function to return the label of the tree.
    def get_plot_label(self, data=None):
        return str(self.state.getLabel(data))

    # str(self) operator overload
    # This function provides a human readable quick information
    # Checks if the state function has an implemented print function before calling it.
    def __str__(self):
        hasStr = True


        result = 'SearchState{rCost='+str(self.rCost)+',hCost='+str(self.hCost)+',invCmp='+ \
            str(self.invertCmp)+',hasPar='+str(self.parent != None)
        try:
            self.state.__str__
        except NameError:
            result += '}'
        else:
            # Has string so it will call the function to get print state
            result += ',state=' + str(self.state) + '}'

        return result

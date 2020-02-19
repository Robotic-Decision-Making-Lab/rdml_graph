# SearchState.py
# Written Ian Rankin - January 2020
#
# A basic state for search functions
# Should be extended for new functionality

from rdml_graph import State

class SearchState(object):
    # Constructor
    # @param state - should be of class State
    def __init__(self, state, rCost=0, hCost=0, parent=None):
        if not isinstance(state, State):
            raise TypeError("Search state given argument state not of type 'State', instead is type: " + str(type(state)))
        self.rCost = rCost # real cost
        self.hCost = hCost # estimated cost
        self.state = state # The actual state
        self.parent = parent # Pointer to parent node of state for finding full path
        self.invertCmp = False # allows inverting invert comparison (say for max heap)

    # successor
    # Gets the successors of the current search state.
    # It calls the states successor function to update the search states
    #
    # @return - a list of SearchState of successors of the current search state.
    def successor(self):
        # consider how to handle the estimated hCost
        return [SearchState(s[0], rCost=self.rCost+s[1], parent=self)  \
                    for s in self.state.successor()]

    # cost
    # returns the combination of real cost and estimated additional cost
    def cost(self):
        return self.rCost + self.hCost

    # getPath
    # This function works its way to up the search tree to the root node
    # to return the list of all states in path.
    def getPath(self):
        if self.parent is None:
            return [self.state]

        return self.parent.getPath() + [self.state]

    ################################ Operator overloads

    # < operator overload
    # Redefine less than operator for heapq
    # Comparison between costs performed
    def __lt__(self, other):
        if self.invertCmp:
            return self.cost() > other.cost()
        else:
            return self.cost() < other.cost()

    # > operator overload
    # Redefine greater than operator for heapq
    # Comparison between costs performed
    def __gt__(self, other):
        if not self.invertCmp:
            return self.cost() > other.cost()
        else:
            return self.cost() < other.cost()

    # == operator overload
    def __eq__(self, other):
        if not isinstance(other, SearchState):
            return False
        return self.cost() == other.cost()

    # str(self) operator overload
    # This function provides a human readable quick information
    # Checks if the state function has an implemented print function before calling it.
    def __str__(self):
        hasStr = True


        result = 'Node{rCost='+str(self.rCost)+',hCost='+str(self.hCost)+',invCmp='+ \
            str(self.invertCmp)+',hasPar='+str(self.parent != None)
        try:
            self.state.__str__
        except NameError:
            result += '}'
        else:
            # Has string so it will call the function to get print state
            result += ',state=' + str(self.state) + '}'

        return result

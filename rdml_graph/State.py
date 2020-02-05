# State.py
# Written Ian Rankin February 2020
#
# A basic class definition of an abstract state for search algorithms
# Each must define its successor function.

class State(object):
    # A default successor Function
    # Abstract function
    #
    # @return - [(successor state, cost)] a list of tuples
    #               with new state and additional cost
    def successor(self):
        return []

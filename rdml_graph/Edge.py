# Edge.py
# Written Ian Rankin October 2019
#
# Rather a directed edge, but called an edge for short.
# Contains the basic implementation for an edge.
# Particular designed to be extendable to allow different information to be
# stored.

class Edge(object):
   # constructor
   # Pass the parent and child nodes directly (not indcies)
   def __init__(self, parent, child, cost = 1.0):
       # should be handed the parent node directly.
       self.p = parent
       self.c = child

       # don't reference directly rather reference the getCost function.
       self.cost = cost

   # get function for cost.
   # This shoud be called rather than directly referencing
   # in case the function is overloaded.
   def getCost(self):
       return self.cost

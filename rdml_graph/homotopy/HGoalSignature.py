
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
# HGoalSignature.py
# Written Ian Rankin - January 2020
#
# A basic structure to handle HGoalSignatures
# Each HGoalSignature is stored as the set of each non-signature obstacles
# or some partial or complete list.




# partial_homology_goal_check
# Checks if the nodes are the same, and checks if the h-signatures fit the
# constraints in HomologySignatureGoal, which allows partial h-signature matches.
# @param n - the input node (Should be a HNode)
# @param data - generic (not used)
# @param goal - the input goal data to the search funcion.
#               MUST match (Node, HomologySignatureGoal type)
def partial_h_goal_check(n, data, goal):
    goalNode = goal[0]
    goalH = goal[1]
    if not isinstance(goalH, HSignatureGoal):
        raise TypeError("partial_h_goal_check given goal which should be (Node, HomologySignatureGoal)")

    return goalH.checkSign(n.h_sign) and goalNode == n.node

# partial_homology_feature_goal
# A function to check for goal states with both homotopy and a topological features.
# This function only checks if a keyword is in the goal set, rather than checking
# for negatives, this can be updated for the future.
# @param n - the input node to check
# @param data - some set of data (not used)
# @param goal - a tuple of (Node, HomologySignatureGoal, names(set), keywords(set))
#
# @return - true if it is a goal state.
def partial_h_feature_goal(n, data, goal):
    goalNode = goal[0]
    goalH = goal[1]
    goalNames = goal[2]
    goalKeywords = goal[3]

    #print(n.node.id)

    if not isinstance(goalH, HSignatureGoal):
        raise TypeError("partial_h_feature_goal given goal which should be (Node, HomologySignatureGoal, set(string), set(string))")

    return goalH.checkSign(n.h_sign) and goalNode == n.node and \
            goalNames <= n.names

# abstract class for an Hsignature goal
class HSignatureGoal(object):

    # checkSign
    # A function to check if the given H signature goal matches the
    def checkSign(self, other):
        raise NotImplementedError()

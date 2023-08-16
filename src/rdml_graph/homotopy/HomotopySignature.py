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
## @package HomotopySignature.py
# Written Ian Rankin - January 2020
#
# A basic structure to handle HSignatures
# These are implemented as the homotopy signature given in:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
#
# Note these are homotopy invariants instead of homology. This implies order
# matters.
# IMPORTANT: Feature numbers must start at 1 and not 0.
# Yes this is weird, but makes computation and storage signicantly faster.



import numpy as np
import copy
from rdml_graph.homotopy import HSignature
from rdml_graph.homotopy.HEdge import HEdge
from rdml_graph.homotopy import HSignatureGoal
from rdml_graph.homotopy.HomologySignature import rayIntersection

# for checking python version (required for hashing function)
import sys

import pdb


## A basic structure to handle HSignatures
# These are implemented as the homotopy signature given in:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
#
# Note these are homotopy invariants instead of homology. This implies order
# matters.
# IMPORTANT: Feature numbers must start at 1 and not 0.
# Yes this is weird, but makes computation and storage signicantly faster.
class HomotopySignature(HSignature):
    ## Constuctor
    # @param numHazards - this is the total number of obstacles the h-signature
    #           needs to keep track of.
    def __init__(self, sign=[]):
        self.sign = copy.copy(sign)
        #self.pythonVer = sys.version_info[0]

    ## edge_cross
    # This function takes the HSignature and the HSign fragment contained in a
    # Homotopy Edge, and adds the edge crossings to the current HSignature.
    # Lazy looping rejection
    # @param edge - a crossing homotopy edge.
    #
    # @return - true if valid edge crossing, false if the crossing is invalid (loop)
    # @post - this objects sign is updated with the given
    def edge_cross(self, edge):
        if not isinstance(edge, HEdge):
            raise TypeError('edge_cross passed an edge which is not of type HEdge')

        if len(edge.HSign) < 1:
            return True

        # self += edge.HSign
        #
        # # check for repeats
        # # Might need a better checking method O(n)
        # for i in range(1, len(self)):
        #     if self.sign[i] == self.sign[i-1]:
        #         return False
        j = 0

        while len(self.sign) > 0 and  j < len(edge.HSign) \
                    and self.sign[-1] == -edge.HSign[j]:
            # remove the canceled sign and increase j
            try:
                self.sign.pop()
            except:
                pdb.set_trace()
            j += 1

        if j < len(edge.HSign):
            if len(self.sign) > 0:
                # check for repeats:
                if self.sign[-1] == edge.HSign[j]:
                    return False
                self.sign += edge.HSign[j:]
            else:
                self.sign = edge.HSign[j:]

        return True

    ## compute_line_segment
    # This function turns the current HSignature into the h signature for a
    # line-segment
    # @param pt_a - the first point of the line segment (numpy)
    # @param pt_a - the second point of the line segment (numpy)
    def compute_line_segment(self, pt_a, pt_b, features, ray_angle=np.pi/2):
        #pdb.set_trace()
        num_features = features.shape[0]
        self.sign = []

        for i in range(num_features):
            sign = rayIntersection(pt_a, pt_b, features[i], ray_angle)
            if sign != 0:
                self.cross(i+1, sign)

        if len(self) > 1:
            # sort the crossings into the correct order.
            # This is done by projecting the given features onto the vector between the parent and child.
            vec = pt_b - pt_a
            projections = [0] * len(self)
            for i in range(len(self)):
                projections[i] = features[abs(self.sign[i])-1].dot(vec)

            _, self.sign = zip(*sorted(zip(projections, self.sign)))
            self.sign = list(self.sign)



    ## cross
    # A function to add a crossing to the HSignature
    # @param id - the id of the feature
    # @param value - the sign of the crossing (+1, 0, -1) 0, makes no sense to be given
    def cross(self, id, value):
        # Bad id just ignore the crosssing
        if id == 0:
            raise ValueError('HomotopySignature id must not be 0')

        if value > 0:
            value = id
        elif value < 0:
            value = -id
        else:
            return

        if len(self.sign) > 0 and self.sign[-1] == -value:
            self.sign.pop()
        else:
            self.sign.append(value)

    ## is_loop
    # This function checkes to see if there is a loop in the homotopy signature.
    # @return true if there is a loop, otherwise false.
    def is_loop(self):
        occurrence = set()
        for cross in self.sign:
            if cross in occurrence:
                return True
            else:
                occurrence.add(cross)
        
        return False

    def copy(self):
        return copy.deepcopy(self)

    ############################## Operator overloading

    def __getitem__(self, key):
        return self.sign[key]

    def __neg__(self):
        sign = self.copy()
        sign.sign.reverse()
        sign.sign = [-s for s in sign.sign]
        return sign

    def __iadd__(self, other):
        # check for canceling signs
        j = 0
        #pdb.set_trace()
        while len(self.sign) > 0 and  j < len(other.sign) \
                    and self.sign[-1] == -other.sign[j]:
            # remove the canceled sign and increase j
            self.sign.pop()
            j += 1

        if j < len(other.sign):
            if len(self.sign) > 0:
                self.sign += other.sign[j:]
            else:
                self.sign = other.sign[j:]
        return self

    def __add__(self, other):
        newSign = self.copy()
        newSign += other
        return newSign

    def __isub__(self, other):
        self += (-other)
        return self

    def __sub__(self, other):
        newSign = self.copy()
        newSign -= other
        return newSign

    ## str(self) operator overload
    # Human readable print output
    def __str__(self):
        return str(self.sign)

    def __hash__(self):
        return hash(tuple(self.sign))
        #if sys.version_info[0] < 3:
        #    return hash(self.sign.data)
        #else:
        #    return hash(self.sign.tobytes())

    ## len(self) operator overload
    def __len__(self):
        return len(self.sign)

    ## == operator overload
    # Function to handle checking for equality between HSignatures
    def __eq__(self, other):
        return self.sign == other.sign

    ## != operator overload
    def __ne__(self, other):
        return not (self == other)


## Signature goal for homotopy.
# Only implemented to
class HomotopySignatureGoal(HSignatureGoal):
    def __init__(self, goal_signature):
        self.goal_sign = goal_signature


    ## checkSign
    # A function to check if the given H signature goal matches the
    def checkSign(self, other):
        if isinstance(other, HomotopySignature):
            return self.goal_sign == other
        else:
            return False













#

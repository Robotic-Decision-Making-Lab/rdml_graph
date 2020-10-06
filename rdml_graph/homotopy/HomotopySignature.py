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
# HomotopySignature.py
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
import sys
#from rdml_graph.homotopy.HomotopyEdge import HomotopyEdge
from rdml_graph.core import Edge

# for checking python version (required for hashing function)
import sys

import pdb

# rayIntersection
# Given a line segment an origin and an angle from the ray, return 0 for
# no intersection, and postive to a crossing in the positive direction, and
# negative for a crossing in the negative direction.
# This code is ported from rdml_utils (Seth McCammon)
# @param pt1 - first point of line segment (numpy 2d)
# @param pt2 - second point of the line segment (numpy 2d)
# @param origin - the origin of the array (numpy 2d)
# @param angle - the angle of the ray. (scalar)
#
# @return - 0 = no intersection, 1 = intersection in positive direction,
#            -1 for negative direction
def rayIntersection(pt1, pt2, origin, angle=np.pi/2):
    rayDir = np.array([np.cos(angle), np.sin(angle)])

    # Create helper vectors
    v1 = origin - pt1
    v2 = pt2 - pt1
    v3 = np.array([-rayDir[1], rayDir[0]]) # perpendicular to rayDir

    # check for parallel lines segment to ray
    if np.dot(v2, v3) == 0:
        return 0 # no intersection

    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)

    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        # there is an intersection decide direction.
        vPt1 = -v1
        #vPt2 = pt2 - origin

        # x coordinate of a transformed point frame.
        xTran1 = -np.dot(vPt1, v3)
        #xTran2 = np.dot(vPt2, v3)
        if xTran1 < 0:
            return 1 # line segment goes left to right across ray.
        else:
            return -1 # line segment goes right to left across ray.
    else:
        # no intersection
        return 0

####################################################################### Need to test Homotopy Edge code.
class HomotopyEdge(Edge):
    # constructor
    # @param parent - the parent node of the edge.
    # @param child - the child node of the edge.
    # @param num_objects - the total number of topological objects.
    # @param cost - the cost of the edge - defaults to 1
    def __init__(self, parent, child, cost=1, features=None,ray_angle=np.pi/2):
        super(HomotopyEdge, self).__init__(parent, child, cost)
        # H-signature fragment (only shows crossing that can occur)

        self.HSign = HomotopySignature([])

        if features is not None:
            self.geo2DHSignCheck(features, ray_angle)


    # geo2DHSignCheck
    def geo2DHSignCheck(self, features, ray_angle=np.pi/2):
        #pdb.set_trace()
        num_features = features.shape[0]

        for i in range(num_features):
            sign = rayIntersection(self.p.pt, self.c.pt, \
                                                features[i], ray_angle)
            if sign != 0:
                self.HSign.cross(i+1, sign)

        if len(self.HSign) > 1:
            # sort the crossings into the correct order.
            # This is done by projecting the given features onto the vector between the parent and child.
            vec = self.c.pt - self.p.pt
            projections = [0] * len(self.HSign)
            for i in range(len(self.HSign)):
                projections[i] = features[abs(self.HSign.sign[i])-1].dot(vec)

            _, self.HSign.sign = zip(*sorted(zip(projections, self.HSign.sign)))


    def __str__(self):
        return 'e(p.id='+str(self.p.id)+',c.id='+str(self.c.id)+',hFrag='+str(self.HSign)+',cost='+str(self.cost)+')'

class HomotopySignature(object):
    # Constuctor
    # @param numHazards - this is the total number of obstacles the h-signature
    #           needs to keep track of.
    def __init__(self, sign):
        self.sign = sign
        #self.pythonVer = sys.version_info[0]

    # edgeCross
    # This function takes the HSignature and the HSign fragment contained in a
    # Homotopy Edge, and adds the edge crossings to the current HSignature.
    # @param edge - a crossing homotopy edge.
    #
    # @return - true if valid edge crossing, false if the crossing is invalid (loop)
    # @post - this objects sign is updated with the given
    def edgeCross(self, edge):
        if not isinstance(edge, HomotopyEdge):
            raise TypeError('edgeCross passed an edge which is not of type HomotopyEdge')

        if len(edge.HSignFrag) < 1:
            return True

        # check if cancel with previous sign
        if -self.sign[-1] == edge.HSign

        ######################################################################################## THIS NEEDS TO BE WRITTEN
        return True


    # cross
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


    def copy(self):
        return copy.deepcopy(self)

    ############################## Operator overloading

    # y = self[idx] operator overload
    # overload square bracket operator
    # Returns element of H-signature
    # @param idx
    # def __getitem__(self, id):
    #     if isinstance(id, slice):
    #         # is a slice, handle slices
    #         return self.sign[id]
    #     else: # Is just a simple index
    #         if id >= len(self) or id < 0:
    #             raise IndexError('H-signature access idx: ' + str(id) + '  with length ' + str(len(self)))
    #         return self.sign[id]
    #
    # def __setitem__(self, key, item):
    #     if item > 1:
    #         raise ValueError('HSignature['+str(key)+'] passed value: '+str(item)+' larger than 1')
    #     elif item < -1:
    #         raise ValueError('HSignature['+str(key)+'] passed value: '+str(item)+' smaller than -1')
    #
    #     self.sign[key] = item

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

    # str(self) operator overload
    # Human readable print output
    def __str__(self):
        return str(self.sign)

    def __hash__(self):
        return hash(tuple(self.sign))
        #if sys.version_info[0] < 3:
        #    return hash(self.sign.data)
        #else:
        #    return hash(self.sign.tobytes())

    # len(self) operator overload
    def __len__(self):
        return len(self.sign)

    # == operator overload
    # Function to handle checking for equality between HSignatures
    def __eq__(self, other):
        return self.sign == other.sign

    # != operator overload
    def __ne__(self, other):
        return not (self == other)


class HSignatureGoal(object):
    def __init__(self, num_objects):
        self.mask = np.zeros(num_objects, dtype=np.bool)
        self.sign = HSignature(num_objects)

    def addConstraint(self, id, value):
        if id >= len(self.mask) or id < 0:
            raise IndexError('H-signature goal access idx: ' + str(id) + '  with length ' + str(len(self.mask)))
        self.mask[id] = 1
        self.sign.cross(id, value)

    def removeConstraint(self, id):
        if id >= len(self.mask) or id < 0:
            raise IndexError('H-signature goal access idx: ' + str(id) + '  with length ' + str(len(self.mask)))
        self.mask[id] = 0
        self.sign.sign[id] = 0

    # checkSign
    # A function to check if the given H signature goal matches the
    def checkSign(self, other):
        return np.all(np.logical_or(np.logical_not(self.mask),\
                                    np.equal(other.sign, self.sign.sign)))














#

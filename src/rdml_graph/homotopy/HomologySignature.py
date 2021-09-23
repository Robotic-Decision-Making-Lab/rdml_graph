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
## @package HomologySignature.py
# Written Ian Rankin - January 2020
#
# A basic structure to handle HomologySignatures
# Each HomologySignature is stored as the set of each non-signature obstacles
# or some partial or complete list.

from __future__ import absolute_import

import numpy as np
from rdml_graph.homotopy import HSignature
from rdml_graph.homotopy import HSignatureGoal
from rdml_graph.homotopy.HEdge import HEdge
import copy

import sys
import pdb

## rayIntersection
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

## Homology signature
# A discrete homology signature.
# It uses the same reference lines for homotopy signautres described by:
# Kim, S., Bhattacharya S., Kumar, V. (2014) Path Planning for a Tethered Mobile Robot
# however, instead keeps an ordered list of crossings. This results in a discretized
# version of winding count used in
# S. Bhattacharya, R. Ghrist, V. Kumar (2015) Persistent Homology for Path Planning
#       in uncertain environments.
class HomologySignature(HSignature):
    ## Constuctor
    # @param numHazards - this is the total number of obstcales the h-signature
    #           needs to keep track of.
    def __init__(self, numHazards):
        self.sign = np.zeros(numHazards, dtype=np.byte)

    ## edge_cross
    # This function takes the HSignature and the HSign fragment contained in a
    # Homotopy Edge, and adds the edge crossings to the current HSignature.
    # @param edge - a crossing homotopy edge.
    #
    # @return - true if valid edge crossing, false if the crossing is invalid (loop)
    # @post - this objects sign is updated with the given
    def edge_cross(self, edge):
        if not isinstance(edge, HEdge):
            raise TypeError('edge_cross passed an edge which is not of type HEdge')
        #pdb.set_trace()

        self.sign += edge.HSign.sign
        if len(self.sign) < 1:
            return True
        elif np.amax(self.sign) > 1 or np.amin(self.sign) < -1:
            return False
        return True

    ## compute_line_segment
    # This function turns the current HSignature into the h signature for a
    # line-segment
    # @param pt_a - the first point of the line segment (numpy)
    # @param pt_a - the second point of the line segment (numpy)
    def compute_line_segment(self, pt_a, pt_b, features, ray_angle=np.pi/2):
        num_features = len(self.sign)

        for i in range(num_features):
            self.sign[i] = rayIntersection(pt_a, pt_b, \
                                                features[i], ray_angle)

    ## cross
    # A function to add a crossing to the HSignature
    def cross(self, id, value):
        # Bad id just ignore the crosssing
        if id >= len(self) or id < 0:
            raise IndexError('H-signature crossing idx: ' + str(id) + '  with length ' + str(len(self)))
        value = max(-1, min(value, 1))
        # Check if the signature contains the id
        self.sign[id] = max(-1, min(value + self.sign[id], 1))

    def copy(self):
        return copy.deepcopy(self)

    ############################## Operator overloading

    ## y = self[idx] operator overload
    # overload square bracket operator
    # Returns element of H-signature
    # @param idx
    def __getitem__(self, id):
        if isinstance(id, slice):
            # is a slice, handle slices
            return self.sign[id]
        else: # Is just a simple index
            if id >= len(self) or id < 0:
                raise IndexError('H-signature access idx: ' + str(id) + '  with length ' + str(len(self)))
            return self.sign[id]

    def __setitem__(self, key, item):
        if item > 1:
            raise ValueError('HSignature['+str(key)+'] passed value: '+str(item)+' larger than 1')
        elif item < -1:
            raise ValueError('HSignature['+str(key)+'] passed value: '+str(item)+' smaller than -1')

        self.sign[key] = item

    def __neg__(self):
        sign = self.copy()
        sign.sign = -sign.sign
        return sign

    def __add__(self, other):
        newSign = self.copy()
        newSign.sign += other.sign
        return newSign

    def __sub__(self, other):
        newSign = self.copy()
        newSign.sign -= other.sign
        return newSign

    ## str(self) operator overload
    # Human readable print output
    def __str__(self):
        return str(self.sign)

    def __hash__(self):
        if sys.version_info[0] > 3:
            return hash(self.sign.data)
        else:
            return hash(self.sign.tobytes())

    ## len(self) operator overload
    def __len__(self):
        return len(self.sign)

    ## == operator overload
    # Function to handle checking for equality between HSignatures
    def __eq__(self, other):
        if len(other) != len(self):
            return False
        return not np.any(np.not_equal(other.sign, self.sign))

    ## != operator overload
    def __ne__(self, other):
        return not (self == other)


## Goal signature for a homology signature
class HomologySignatureGoal(HSignatureGoal):
    def __init__(self, num_objects):
        self.mask = np.zeros(num_objects, dtype=np.bool)
        self.sign = HomologySignature(num_objects)

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

    ## checkSign
    # A function to check if the given H signature goal matches the
    def checkSign(self, other):
        return np.all(np.logical_or(np.logical_not(self.mask),\
                                    np.equal(other.sign, self.sign.sign)))

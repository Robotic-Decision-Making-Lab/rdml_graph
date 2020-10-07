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
# HSignature.py
# Written Ian Rankin - January 2020
#
# A basic structure to handle HSignatures
# Each HSignature is stored as the set of each non-signature obstacles
# or some partial or complete list.

import numpy as np
import copy
import sys


# for checking python version (required for hashing function)
import sys

class HSignature(object):

    # @overide
    # @param edge - the HEdge that should be added to the H signature to update
    #               to a new state
    # @return - True if valid edge crossing, false if the crossing is invalid (loop)
    def edge_cross(self, edge):
        raise NotImplementedError()

    # cross
    # A function to add a crossing to the HSignature
    # @param id - the id of the feature
    # @param value - the sign of the crossing (+1, 0, -1) 0, makes no sense to be given
    def cross(self, id, value):
        raise NotImplementedError()

    # compute_line_segment
    # This function turns the current HSignature into the h signature for a
    # line-segment
    # @param pt_a - the first point of the line segment (numpy)
    # @param pt_a - the second point of the line segment (numpy)
    def compute_line_segment(self, pt_a, pt_b, features, ray_angle=np.pi/2):
        raise NotImplementedError()

    # ensure, correct copying of the signatures
    def copy(self):
        raise NotImplementedError()

    # concatination
    # Note concationation for HSignatures may not be commutative (a+b)!=(b+a)
    # For Homotopy signatures (a+b)!=(b+a)
    # For Homology signatures a+b=b+a
    def __add__(self, other):
        raise NotImplementedError()

    # inverse concatination
    def __sub__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()



















#

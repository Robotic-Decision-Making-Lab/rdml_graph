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
from rdml_graph.homotopy.HomotopyEdge import HomotopyEdge


# for checking python version (required for hashing function)
import sys

class HSignature(object):
    # Constuctor
    # @param numHazards - this is the total number of obstcales the h-signature
    #           needs to keep track of.
    def __init__(self, numHazards):
        self.sign = np.zeros(numHazards, dtype=np.byte)
        self.pythonVer = sys.version_info[0]

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
        self.sign += edge.HSignFrag
        if len(self.sign) < 1:
            return True
        elif np.amax(self.sign) > 1 or np.amin(self.sign) < -1:
            return False
        return True


    # cross
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

    # y = self[idx] operator overload
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

    # str(self) operator overload
    # Human readable print output
    def __str__(self):
        return str(self.sign)

    def __hash__(self):
        if self.pythonVer < 3:
            return hash(self.sign.data)
        else:
            return hash(self.sign.tobytes())

    # len(self) operator overload
    def __len__(self):
        return len(self.sign)

    # == operator overload
    # Function to handle checking for equality between HSignatures
    def __eq__(self, other):
        if len(other) != len(self):
            return False
        return not np.any(np.not_equal(other.sign, self.sign))

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

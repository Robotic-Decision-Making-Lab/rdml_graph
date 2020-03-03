# HSignature.py
# Written Ian Rankin - January 2020
#
# A basic structure to handle HSignatures
# Each HSignature is stored as the set of each non-signature obstacles
# or some partial or complete list.

import numpy as np
import copy

from .HomotopyEdge import HomotopyEdge

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
        if np.amax(self.sign) > 1 or np.amin(self.sign) < -1:
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

    # checkSign
    # A function to check if the given H signature goal matches the
    def checkSign(self, other):
        return np.all(np.logical_or(np.logical_not(self.mask),\
                                    np.equal(other.sign, self.sign.sign)))












#

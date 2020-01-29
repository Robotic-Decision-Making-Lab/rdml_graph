# HSignature.py
# Written Ian Rankin - January 2020
#
# A basic structure to handle HSignatures
# Each HSignature is stored as the set of each non-signature obstacles
# or some partial or complete list.

import numpy as np
import copy

class HSignature(object):
    # Constuctor
    # @param numHazards - this is the total number of obstcales the h-signature
    #           needs to keep track of.
    def __init__(self, numHazards):
        self.sign = np.zeros(numHazards)

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
        return copy.copy(self)

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

    # str(self) operator overload
    # Human readable print output
    def __str__(self):
        return str(self.sign)


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

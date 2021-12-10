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
# other_utilities.py
# Written Ian Rankin April 2021
#
# A set of random utilities that didn't have another home.

import time

## get_indicies_from
# This function returns the elements in the parallel lists from the indicies
# Just a convience function that means I don't have to think about it
# @param indicies - the list or numpy array of indicies to downselect from
# @param *parallel_lists - the list of lists to downselect elements from
#
# @return - the tuple of down selected indicies
def get_indicies_from(indicies, *parallel_lists):
    down_selected = [[lis[a] for a in indicies] for lis in parallel_lists]

    return tuple(down_selected)

## str_timestamp
# creates a string versiion of a timestamp.
# Designed to be used to save files without having to worry about conflicts
# between runs.
#
# @return the timestamp as a string.
def str_timestamp():
    t = time.localtime()
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', t)
    return timestamp

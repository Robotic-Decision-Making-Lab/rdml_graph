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
## @package ParetoFront.py
# Written Ian Rankin - February 2021
#
#

import numpy as np
import copy
import pdb

class ParetoFront:
    def __init__(self, reward_dim=3, alloc_size=30):
        self.front = np.empty((alloc_size, reward_dim))
        self.front_val = np.empty(alloc_size, dtype=object)
        self.size = 0


    ## check_and_add
    # quickly checks if element should be in the pareto front and adds it to
    # the front if it should.
    # @param r - the input reward vector
    # @param n - the object associated with the reward vector
    def check_and_add(self, r, n):
        is_efficient = self.check_efficient(r)
        if is_efficient:
            self.add(r, n)

        return is_efficient

    # get
    # this function returns the front to the user
    #
    # @return front, front_vals
    def get(self):
        return self.front[:self.size], self.front_val[:self.size]

    def get_random(self):
        rand_idx = np.random.randint(0, self.size)
        return self.front[rand_idx], self.front_val[rand_idx]

    # check if point is efficient.
    # @param r - the input reward vector
    #
    # @return -
    def check_efficient(self, r):
        if self.size == 0:
            return True

        # check if the point is efficient
        is_efficient = (np.any(self.front[:self.size] < r, axis=1)).all()
        #is_efficient = np.all(
        #                    np.logical_and(np.any(self.front[:self.size] < r, axis=1), \
        #                        np.np.equal(self.front[:self.size], r), axis=1)))

        #strictly_better = self.front[:self.size] < r
        #same_dim = np.equals(self.front[:self.size], r)

        return is_efficient





    def add(self, r, n):
        # check if size is not zero
        #print(r)
        #print(n)
        if self.size > 0:

            #is_dominated = np.all(self.front[:self.size] <= r, axis=1)
            strictly_better = self.front[:self.size] < r
            eq_dim = np.equal(self.front[:self.size], r)

            is_dominated = np.logical_and( \
                                np.all(np.logical_or(strictly_better, eq_dim), axis=1), \
                                np.any(strictly_better, axis=1))

            # new indicies
            lower_idx = self.size - 1
            cur_idx = 0
            indicies = np.empty(self.size, dtype=np.int32)

            #pdb.set_trace()
            ##################### TODO, this needs to be vectorized to be faster.
            for i in range(self.size):
                if is_dominated[i]:
                    indicies[i] = lower_idx
                    lower_idx -= 1
                else:
                    indicies[i] = cur_idx
                    cur_idx += 1
            #################### end what needs to be vectorized

            # remove dominated
            #pdb.set_trace()
            #print(self.front[:self.size])
            #print(indicies)
            # TODO: this op is slow
            self.front[indicies] = copy.copy(self.front[:self.size])
            self.front_val[indicies] = copy.copy(self.front_val[:self.size])
            # TODO: end operation that may be slow
            self.size -= np.sum(is_dominated)


        if self.size == self.front.shape[0]:
            # reallocate front
            tmp = self.front
            tmp_val = self.front_val
            self.front = np.empty((self.front.shape[0]*2, self.front.shape[1]))
            self.front_val = np.empty(self.size*2, dtype=object)
            self.front[:self.size] = tmp
            self.front_val[:self.size] = tmp_val

        # add reward vector
        self.front[self.size] = r
        self.front_val[self.size] = n
        self.size += 1


try:
    from numba import jit

    # same function as below, but written with numba for speed.
    # also written using copilot, and uh it seems to work.
    # It at least passed all of the tests I wrote.
    @jit(nopython=True)
    def get_pareto(values):
        n_points = values.shape[0]
        is_efficient = np.arange(n_points, dtype=np.int32)
        values_work = values.copy()
        next_point_index = 0
        
        while next_point_index < len(values_work):
            nondominated_point_mask = np.zeros(values_work.shape[0], dtype=np.bool_)
            for i in range(values_work.shape[0]):
                for j in range(values_work.shape[1]):
                    if values_work[i, j] > values_work[next_point_index, j]:
                        nondominated_point_mask[i] = True
                        break
            nondominated_point_mask[next_point_index] = True
            
            # Count non-dominated points
            count = 0
            for i in range(len(nondominated_point_mask)):
                if nondominated_point_mask[i]:
                    count += 1
            
            # Create new arrays
            new_is_efficient = np.empty(count, dtype=np.int32)
            new_values = np.empty((count, values_work.shape[1]), dtype=values_work.dtype)
            
            # Copy non-dominated elements
            idx = 0
            for i in range(len(nondominated_point_mask)):
                if nondominated_point_mask[i]:
                    new_is_efficient[idx] = is_efficient[i]
                    for j in range(values_work.shape[1]):
                        new_values[idx, j] = values_work[i, j]
                    idx += 1
            
            is_efficient = new_is_efficient
            values_work = new_values
            
            # Calculate next_point_index
            count_before = 0
            for i in range(min(next_point_index, len(nondominated_point_mask))):
                if nondominated_point_mask[i]:
                    count_before += 1
            next_point_index = count_before + 1
        
        return is_efficient

except ImportError:
    ## get_pareto
    # This function returns the indicies of the pareto optimal values in values
    # Code taken directly from this stackoverflow post for the function
    # is_pareto_efficient by user 'Peter'
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    # @param values - a numpy array of n values with k dimmensions numpy(n, k)
    #
    # @return - numpy array of indicies
    def get_pareto(values, return_mask=False):
        is_efficient = np.arange(values.shape[0], dtype=int)
        n_points = values.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(values):
            nondominated_point_mask = np.any(values>values[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            values = values[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient



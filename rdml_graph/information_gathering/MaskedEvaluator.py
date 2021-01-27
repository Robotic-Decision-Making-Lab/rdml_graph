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
# MaskedEvaluator.py
# Written Ian Rankin January 2021
#
# Similar to the PathEvaluatorWithRadius in Evaluator.py, this code is also path
# evaluator, that only works with radius with masked path evaluation.
# This is much faster, and should be used over path evaluator with radius if
# helped.
#
# TODO: Make PathEvaluatorWithRadius use this evaluator if the method is radius?

import numpy as np
from rdml_graph.information_gathering import PathEvaluator
import pdb


# parameterizeLine
# parameterize line using the equation ax + by + c = 0
# @param pt1 - the first point
# @param pt2 - the second point
#
# @return a,b,c
def parameterizeLine(pt1, pt2):
    if pt1[0] - pt2[0] == 0:
        a = 1
        b = -a * (pt1[0] - pt2[0]) / (pt1[1] - pt2[1])
    else:
        b = 1
        a = -b * (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])

    c = -a * pt1[0] - b * pt1[1]

    return a,b,c


class MaskedEvaluator(PathEvaluator):

    # constructor
    # @param info_field - numpy array (x,y,channels) OR (x,y)
    # @param x_ticks - (should be even ticks.)
    # @param y_ticks - (should be even ticks.)
    # @param radius - the radius of evaluation
    # @param channels - a sequence object of indcies to output from the info_field
    #         , if left as None all indcies are ignored. (if info_field has only)
    #          one dimminsion, then this argument is ignored.
    def __init__(self, info_field, x_ticks, y_ticks, radius, channels=None):
        # handle single dimminsion info_fields for backwards compatability
        if len(info_field.shape) == 2:
            self.info_field = info_field[:,:, np.newaxis]
            self.chan = np.array([1], dtype=np.int16)
        else:
            self.info_field = info_field

            # handle channels not being used.
            if channels is None:
                self.chan = np.arange(0,info_field.shape[2],1, dtype=np.int16)
            else:
                self.chan = channels

        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.radius = radius




    def gen_slices_along_x(self, x, min_pt, max_pt, line_low_loc, line_high_loc, line_param, isTop):
        a,b,c = line_param

        if isTop:
            y_mins = np.where(np.logical_and(x >= line_low_loc, x <= line_high_loc), -(a*x + c) / b, \
                    np.where(x < line_low_loc, min_pt[1] + np.sqrt(self.radius**2 - (min_pt[0]-x)**2), \
                                       max_pt[1] + np.sqrt(self.radius**2 - (max_pt[0]-x)**2)))
        else:
            y_mins = np.where(np.logical_and(x >= line_low_loc, x <= line_high_loc), -(a*x + c) / b, \
                    np.where(x < line_low_loc, min_pt[1] - np.sqrt(self.radius**2 - (min_pt[0]-x)**2), \
                                       max_pt[1] - np.sqrt(self.radius**2 - (max_pt[0]-x)**2)))

        return y_mins


    def getSegmentAlongX(self, pt1, pt2, cur_mask):
        scores = np.zeros(len(self.chan))



        dir = pt1 - pt2 # find the norm of the vector
        dir = dir / np.linalg.norm(dir)
        perp = np.array([-dir[1], dir[0]]) # forced to be vertical
        if perp[1] < 0:
            perp = -perp

        #end_pt1 = -dir*self.radius + pt1
        #end_pt2 = dir*self.radius + pt2
        if pt1[0] < pt2[0]:
            min_pt = pt1
            max_pt = pt2
        else:
            min_pt = pt2
            max_pt = pt1

        upper_min_per = min_pt + perp*self.radius
        upper_max_per = max_pt + perp*self.radius
        lower_min_per = min_pt - perp*self.radius
        lower_max_per = max_pt - perp*self.radius





        upper_param = parameterizeLine(upper_min_per, upper_max_per)
        #a_cen,b_cen,c_cen = parameterizeLine(pt1, pt2)
        #a_low,b_low,c_low = parameterizeLine(lower_min_per, lower_max_per)
        low_param = parameterizeLine(lower_min_per, lower_max_per)


        min_x = min_pt[0] - self.radius
        max_x = max_pt[0] + self.radius
        x_range = np.arange(min_x, max_x+1, 1)

        min_y = self.gen_slices_along_x(x_range, min_pt, max_pt, \
                                    line_low_loc=lower_min_per[0], \
                                    line_high_loc=lower_max_per[0], \
                                    line_param=low_param, \
                                    isTop = False)
        max_y = self.gen_slices_along_x(x_range, min_pt, max_pt, \
                                    line_low_loc=upper_min_per[0], \
                                    line_high_loc=upper_max_per[0], \
                                    line_param=upper_param, \
                                    isTop = True)

        #pdb.set_trace()
        import matplotlib.pyplot as plt
        plt.plot(x_range,min_y)
        plt.plot(x_range,max_y)
        plt.plot(np.array(pt1[0],pt2[0]), np.array(pt1[1],pt2[1]), color='green')
        plt.show()



        for x in x_range:
            min_y = int(floor(min_y[x]))
            max_y = int(ceil(max_y[x]))

            # (condition, true_selector, false_selector)
            scores_raw = np.where(cur_mask[x,min_y:max_y]==1, \
                        np.zeros((1,max_y - min_y, len(self.chan))), \
                        self.info_field[x, min_y:max_y, self.chan])

            scores += np.sum(scores_raw, axis=[0,1])
            cur_mask[x, min_y:max_y] = 1

        return scores


    # @override
    # getScore, gets the score of path given within the budget
    # @param path - the path as 2d numpy array (n x 2)
    # @param budget - the budget of the path, (typically path length)
    def getScore(self, path, budget=float('inf')):
        # force path along budget (TODO)

        mask = np.zeros(self.info_field.shape, dtype=np.int8)
        scores = np.zeros(len(self.chan))

        for i in range(1, len(path)):
            pt1 = path[i-1]
            pt2 = path[i]

            # TODO convert points to x and y ticks.
            scores += self.getSegmentAlongX(pt1, pt2, mask)

        return scores


























































#

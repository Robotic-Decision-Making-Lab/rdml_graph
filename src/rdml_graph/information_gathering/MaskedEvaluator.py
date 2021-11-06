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
## @package MaskedEvaluator.py
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
from rdml_graph.information_gathering import applyBudget
import math

import pdb


## parameterizeLine
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

    ## constructor
    # @param info_field - numpy array (x,y,channels) OR (x,y)
    # @param x_ticks - (should be even ticks.)
    # @param y_ticks - (should be even ticks.)
    # @param radius - the radius of evaluation
    # @param normalize - [opt] this sets if the cost function is attempted to normalize to [0,1]
    #           using radius and budet and expected value of a pixel - Defaults to true
    # @param expected_val - [opt] the exepected value of a single grid square.
    #           defaults to 0.5 for the pixel average.
    # @param channels - [opt] a sequence object of indcies to output from the info_field
    #         , if left as None all indcies are ignored. (if info_field has only)
    #          one dimminsion, then this argument is ignored.
    def __init__(self, info_field, x_ticks, y_ticks, radius, budget=float('inf'),\
                    normalize=True, expected_val=None, channels=None):
        # handle single dimminsion info_fields for backwards compatability
        if len(info_field.shape) == 2:
            self.info_field = info_field[:,:, np.newaxis]
            self.chan = np.array([0], dtype=np.int16)
        else:
            self.info_field = info_field

            # handle channels not being used.
            if channels is None:
                self.chan = np.arange(0,info_field.shape[2],1, dtype=np.int16)
            else:
                self.chan = channels

        self.x_ticks = x_ticks
        self.x_scale = x_ticks[1] - x_ticks[0]

        self.y_ticks = y_ticks
        self.y_scale = y_ticks[1] - y_ticks[0]

        min_scale = min(self.x_scale, self.y_scale)
        if radius < (min_scale/2):
            radius = min_scale/2
        self.radius = radius
        self.normalize=normalize

        if expected_val is None:
            #self.expected_val = np.ones(len(self.chan))
            self.expected_val = np.mean(self.info_field, axis=(0,1))
        else:
            self.expected_val = expected_val

        self.budget = budget
        if budget == float('inf'):
            self.scales = np.ones(len(self.chan))
        else:
            #pdb.set_trace()
            #scaled_radius = radius / np.mean([self.x_scale, self.y_scale])
            #budget_grid = budget / np.mean([self.x_scale, self.y_scale])
            length_grid = (budget+radius*2)  / self.y_scale
            width_grid = (radius*2) / self.x_scale
            self.scales = 1 / (self.expected_val*(length_grid*width_grid))
            #self.scales = 1 / (self.expected_val*(budget_grid+scaled_radius*2)*scaled_radius*2)


    def gen_slices_along_x(self, x, min_pt, max_pt, line_low_loc, line_high_loc, line_param, isTop):
        a,b,c = line_param

        if isTop:
            y_mins = np.where(np.logical_and(x > line_low_loc, x < line_high_loc), -(a*x + c) / b, \
                    np.where(x < line_low_loc, min_pt[1] +  \
                                            np.sqrt(np.maximum(self.radius**2 - (min_pt[0]-x)**2, 0)), \
                                       max_pt[1] + \
                                            np.sqrt(np.maximum(self.radius**2 - (max_pt[0]-x)**2, 0))))
        else:
            y_mins = np.where(np.logical_and(x > line_low_loc, x < line_high_loc), -(a*x + c) / b, \
                    np.where(x < line_low_loc, min_pt[1] -  \
                                            np.sqrt(np.maximum(self.radius**2 - (min_pt[0]-x)**2, 0)), \
                                       max_pt[1] - \
                                            np.sqrt(np.maximum(self.radius**2 - (max_pt[0]-x)**2, 0))))

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

        # if pt1[0] == pt2[0]:
        #     pdb.set_trace()

        # upper_line = np.append(upper_min_per[np.newaxis,:], upper_max_per[np.newaxis, :], axis=0)
        # line = np.append(pt1[np.newaxis,:], pt2[np.newaxis, :], axis=0)
        # lower_line = np.append(lower_min_per[np.newaxis,:], lower_max_per[np.newaxis, :], axis=0)
        #
        #
        # import matplotlib.pyplot as plt
        #plt.figure()

        #plt.scatter(upper_min_per[0][np.newaxis], upper_min_per[1][np.newaxis])
        #plt.scatter(upper_max_per[0][np.newaxis], upper_max_per[1][np.newaxis])
        #plt.scatter(lower_min_per[0][np.newaxis], lower_min_per[1][np.newaxis])
        #plt.scatter(lower_max_per[0][np.newaxis], lower_max_per[1][np.newaxis])
        # plt.plot(upper_line[:,0], upper_line[:,1])
        # plt.plot(line[:,0], line[:,1])
        # plt.plot(lower_line[:,0], lower_line[:,1])
        # plt.xlim([self.x_ticks[0], self.x_ticks[-1]])
        # plt.ylim([self.y_ticks[0], self.y_ticks[-1]])


        upper_param = parameterizeLine(upper_min_per, upper_max_per)
        #a_cen,b_cen,c_cen = parameterizeLine(pt1, pt2)
        #a_low,b_low,c_low = parameterizeLine(lower_min_per, lower_max_per)
        low_param = parameterizeLine(lower_min_per, lower_max_per)


        min_x = max(min_pt[0] - self.radius, self.x_ticks[0])
        max_x = min(max_pt[0] + self.radius, self.x_ticks[-1]+self.x_scale)

        min_x_idx = math.ceil((min_x - self.x_ticks[0]) / self.x_scale)
        max_x_idx = math.ceil((max_x - self.x_ticks[0]) / self.x_scale)
        # ensure the indicies are within the information field
        min_x_idx = max(0, min_x_idx)
        max_x_idx = min(self.info_field.shape[0], max_x_idx)
        if min_x_idx > self.info_field.shape[0]:
            return np.zeros(len(self.chan))
        elif max_x_idx < 0:
            return np.zeros(len(self.chan))

        x_range = np.arange((min_x_idx*self.x_scale)+self.x_ticks[0], \
                ((max_x_idx+1)*self.x_scale)+self.x_ticks[0]-0.01, self.x_scale)


        min_y_arr = self.gen_slices_along_x(x_range, min_pt, max_pt, \
                                    line_low_loc=lower_min_per[0], \
                                    line_high_loc=lower_max_per[0], \
                                    line_param=low_param, \
                                    isTop = False)
        max_y_arr = self.gen_slices_along_x(x_range, min_pt, max_pt, \
                                    line_low_loc=upper_min_per[0], \
                                    line_high_loc=upper_max_per[0], \
                                    line_param=upper_param, \
                                    isTop = True)

        #pdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.plot(x_range,min_y)
        # plt.plot(x_range,max_y)
        # plt.plot(np.array(pt1[0],pt2[0]), np.array(pt1[1],pt2[1]), color='green')
        # plt.show()

        for i, x in enumerate(range(min_x_idx, max_x_idx)):
            #print('min_y_arr[i]: ' + str(min_y_arr[i])+ ' max_y_arr[i]: '+str(max_y_arr[i]))
            min_y = math.ceil((max(min_y_arr[i], self.y_ticks[0])-self.y_ticks[0]) / self.y_scale)
            max_y = math.ceil((min(max_y_arr[i], self.y_ticks[-1]+self.y_scale)\
                                -self.y_ticks[0]) / self.y_scale)
            min_y = max(min_y, 0)
            max_y = min(max_y, self.info_field.shape[1])
            if min_y > self.info_field.shape[1]:
                continue
            elif max_y < 0:
                continue

            #print('min_y: '+str(min_y)+' max_y: '+str(max_y))
            # min_max_pts = np.array([[x, min_y_arr[i]], [x, max_y_arr[i]]])
            # plt.scatter(min_max_pts[:,0], min_max_pts[:,1])


            shaped_mask = np.repeat(cur_mask[x,min_y:max_y, np.newaxis]==1, \
                                    len(self.chan), \
                                    axis=1)

            # (condition, true_selector, false_selector)
            scores_raw = np.where(shaped_mask, \
                        np.zeros((max_y - min_y, len(self.chan))), \
                        self.info_field[x, min_y:max_y, self.chan].transpose())


            scores = self.determineScore(shaped_mask, scores, x, min_y, max_y)
            cur_mask[x, min_y:max_y] = 1

        #print(cur_mask)

        return scores

    def determineScore(self, shaped_mask, scores, x, min_y, max_y):
        scores_raw = np.where(shaped_mask, \
                    np.zeros((max_y - min_y, len(self.chan))), \
                    self.info_field[x, min_y:max_y, self.chan].transpose())

        scores += np.sum(scores_raw, axis=0)
        return scores


    ## @override
    # getScore, gets the score of path given within the budget
    # @param path - the path as 2d numpy array (n x 2)
    # @param budget - the budget of the path, (typically path length)
    def getScore(self, path, budget=None, return_mask=False):
        if budget is None:
            budget = self.budget
        path, length = applyBudget(path, budget)

        mask = np.zeros(self.info_field.shape[0:2], dtype=np.int8)
        scores = np.zeros(len(self.chan))

        for i in range(1, len(path)):
            pt1 = path[i-1]
            pt2 = path[i]

            scores += self.getSegmentAlongX(pt1, pt2, mask)

        #print(mask)
        if return_mask:
            return scores * self.scales, mask

        return scores * self.scales

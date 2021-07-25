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
## @package StochasticOptimizer.py
# Written Dylan Jones, revised by Ian Rankin April 2020
#
# Path of waypoints optimized by performing stochastic gradient ascent.
# Gradients caluclated by perturbating the path and approximating based on pertubations.

import numpy as np
import random, tqdm
import pdb

from scipy.interpolate import RegularGridInterpolator

from rdml_graph.information_gathering import PathEvaluator

import shapely.geometry as geo
import shapely.ops as ops

## Class for using stocastic optimization for updating a path
#   - pert_size:    [x_size, y_size] is the standard deviation for pertubation in the x and y
#   - num_perts:    number of pertubation to use when calculating the gradient
#   - num_its:      number of iterations before stopping optimization
#   - h:            controls the size of the gradient update, larger mean bigger steps
class StochasticOptimizer(object):

    # @param yaml_optimizer - a dictionary of parameters for the Stoachastic Optimizer
    # @param bounds - TBD
    def __init__(self, yaml_optimizer, bounds):
        # pert_size=[1,1], num_perts=25, num_its=100, h=1):
        self.pert_size = [yaml_optimizer['perturbation_variance_x'], yaml_optimizer['perturbation_variance_y']]
        self.num_perts = yaml_optimizer['number_perturbations']
        self.num_its = yaml_optimizer['number_iterations']
        self.h = yaml_optimizer['gradient_step_weight']
        self.cooling_rate = yaml_optimizer['cooling_rate']
        #self.homotopy_alpha = yaml_optimizer['homotopy_alpha']
        self.verbose = yaml_optimizer['verbose']


        self.bounds = bounds

    # Main function call, given a path and an environment it optimizes the path
    # Expects that the environment is a class supporting a getScore(path) call
    # with path as a numpy array [[x1, y1] ... [xn, yn]]
    # @param path - numpy array [[x1,y1], ... [nx, yn]]
    # @param reward_func, a reward function of the type (sequence, budget, data)
    #                   the reward function is the same format as MCTS.
    # @param budget the required budget of the optimizer
    # @param data - any required data for the optimization process.
    def optimize(self, path, rewardFunc, budget, data=None):
        self.reference_path = path

        # env.reference_h_sig = HSignature.fromPath(path, env.rep_pts)

        np_path = path#np.vstack([x.npArray() for x in path])
        p_length = len(path)

        for ii in tqdm.tqdm(range(self.num_its)):

            if self.verbose:
                print("Percent done: %.2f" % ((float(ii) / float(self.num_its)) * 100))

            #pert_order = range(1,p_length)
            pert_order = np.arange(1,p_length)
            random.shuffle(pert_order)

            for index in pert_order:
                perts = self.genPerts(p_length)
                perts[0] = np.array([0,0])
                scores = self.scorePerts(np_path, index, perts, rewardFunc, budget, data)
                grad = self.calGrad(scores, perts)
                np_path[index] = np_path[index] + grad * (self.cooling_rate ** ii)
                if np.isnan(np_path).any():
                    pdb.set_trace()

        #out_path = [Location(xlon=x[0], ylat=x[1]) for x in np_path]

        return np_path

    # # Main function call, given a path and an environment it optimizes the path
    # # Expects that the environment is a class supporting a getScore(path) call with path as a numpy array [[x1, y1] ... [xn, yn]]
    # def optimizeEndpointConstrained(self, path, env, budget):
    #     self.reference_path = path
    #
    #     # env.reference_h_sig = HSignature.fromPath(path, env.rep_pts)
    #
    #     np_path = np.vstack([x.npArray() for x in path])
    #     p_length = len(path)
    #
    #     for ii in range(self.num_its):
    #         pert_order = range(1,p_length-1)
    #         random.shuffle(pert_order)
    #
    #         for index in pert_order:
    #             perts = self.genPerts(p_length)
    #             perts[0] = np.array([0,0])
    #             scores = self.scorePerts(np_path, index, perts, env, budget)
    #             grad = self.calGrad(scores, perts)
    #             np_path[index] = np_path[index] + grad * (self.cooling_rate ** ii)
    #             if np.isnan(np_path).any():
    #                 pdb.set_trace()
    #
    #     out_path = [Location(xlon=x[0], ylat=x[1]) for x in np_path]
    #
    #     assert path[-1] == out_path[-1]
    #
    #     return out_path


    # Function to generate the pertubations based on the given pertubation size
    def genPerts(self, path_length):
        perts = np.random.multivariate_normal([0,0], [[self.pert_size[0], 0],[0, self.pert_size[1]]], size=(self.num_perts))
        return perts

    # Function to calculate a score for each of the pertubations
    def scorePerts(self, path, index, perts, rewardFunc, budget, data):
        scores = []

        for pert_idx, pert in enumerate(perts):
            pert_path = np.copy(path)
            a = pert_path[index] + pert

            shapelyPt = geo.Point(a)
            if not self.bounds.contains(shapelyPt):
                #modified_pert = self.world_est.getClosestInBounds(Location(xlon=a[0], ylat=a[1])) - Location(xlon=pert_path[index][0], ylat=pert_path[index][1])
                #pdb.set_trace()
                modified_pert, _ = ops.nearest_points(self.bounds, shapelyPt)
                modified_pert = np.array([modified_pert.x, modified_pert.y]) - pert_path[index]
                a = pert_path[index] + modified_pert
                #a = modified_pert
                perts[pert_idx,:] = modified_pert

            pert_path[index] = a

            if np.isnan(pert_path).any():
                pdb.set_trace()

            scores.append(rewardFunc(pert_path, budget, data))

        return scores

    # Function to calculate the gradient based upon the given scores
    def calGrad(self, scores, perts):

        grad = np.array([0,0])

        max_s = np.amax(scores)
        min_s = np.amin(scores)

        for ii, s in enumerate(scores):

            if max_s != min_s:
                w = np.exp(self.h * ( (s - min_s) / (max_s - min_s) ) )
            else:
                w = 1

            grad = grad + (w / self.num_perts) * perts[ii]

        return grad

    # Function to get the reward for a path from an environment
    # def rewardFun(self, sub_path, env, budget):
    #     return env.getScore(sub_path, budget)[0]  #+ self.homotopy_alpha*env.getHomotopyScore(query_loc_path)


    def setPertSize(self, pert_size):
        self.pert_size = pert_size

    def setNumPerts(self, num_perts):
        self.num_perts = num_perts

    def setNumIts(self, num_its):
        self.num_its = num_its

    def setH(self, h):
        self.h = h


#
# def main():
#     opt = StochasticOptimizer(num_its=10, num_perts=5)
#
#     n_ticks = 51
#
#     x_ticks = np.linspace(-25., 25., n_ticks)
#     y_ticks = np.linspace(-25., 25., n_ticks)
#     info_field = np.random.random((len(x_ticks), len(y_ticks)))
#     cost_field = np.random.random((len(x_ticks), len(y_ticks)))
#     infoInterpFn = RegularGridInterpolator([x_ticks, y_ticks], info_field)
#     costInterpFn = RegularGridInterpolator([x_ticks, y_ticks], cost_field)
#     yaml_eval = {'step_size':.2}
#     path = [Location(0, 0),
#             Location(1, 1),
#             Location(0, 1)]
#
#
#     evaluator = PathEvaluator(infoInterpFn, costInterpFn, yaml_eval)
#
#     for item in path:
#         print(item)
#
#     p_out = opt.optimize(path,evaluator)
#
#     for item in p_out:
#         print(item)
#
# if __name__ == '__main__':
#     main()

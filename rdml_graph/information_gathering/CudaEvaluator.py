# CudaEvaluator.py
# Written Ian Rankin - July 2021
#
# @file CudaEvaluator
# A cuda evaluator for path integrals


import numpy as np
from numba import cuda
from rdml_graph.information_gathering import PathEvaluator
import math

import pdb


RADIUS_DIST = 0


## CudaEvaluator
# an evaluator which finds the min distance to a point and applies a function to
# it.
# This allows each pixel to be it's own thread individually
class CudaEvaluator(PathEvaluator):

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
                    normalize=True, expected_val=None, channels=None, max_num_paths=30):
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
            num_pixels = (budget+radius*2)*2*radius
            self.scales = 1 / (self.expected_val*num_pixels*2)

        self.reward_arr = np.empty((len(self.chan), self.info_field.shape[0]*self.info_field.shape[1]))
        self.multi_reward = np.empty((max_num_paths, \
                                      self.info_field.shape[0], self.info_field.shape[1], \
                                      len(self.chan)))

        # device memory
        self.info_field = cuda.to_device(self.info_field)
        self.reward_arr = cuda.to_device(self.reward_arr)
        self.multi_reward = cuda.to_device(self.multi_reward)
        self.chan = cuda.to_device(self.chan)
        self.x_ticks = cuda.to_device(self.x_ticks)
        self.y_ticks = cuda.to_device(self.y_ticks)





    ## @override
    # getScoreMulti, gets the score of path given within the budget
    # @param path - the path as 2d numpy array (n x 2)
    # @param budget - the budget of the path, (typically path length)
    def getScoreMulti(self, paths, lengths, budget=float('inf')):
        # force path along budget (TODO)
        threads_per_block = 32
        blocks = math.ceil(len(lengths)*self.info_field.shape[0]*self.info_field.shape[1] / threads_per_block)

        #reward_arr = np.empty((self.info_field.shape[0], self.info_field.shape[1], len(self.chan)))
        #reward_arr = np.empty((len(self.chan), self.info_field.shape[0]*self.info_field.shape[1]))
        #reward_arr = np.ascontiguousarray(reward_arr)
        paths_cuda = cuda.to_device(paths)

        num_rewards = self.info_field.shape[0]*self.info_field.shape[1]
        num_threads = 1024
        sum_per_thread = math.ceil(num_rewards / num_threads)

        scores = np.empty((lengths.shape[0], len(self.chan)))

        if sum_per_thread < 8:
            sum_per_thread = 8
            num_threads = math.ceil(num_rewards / 8)

        reward_sum_log = int(np.ceil(np.log(num_rewards) / np.log(sum_per_thread)))
        #reward_sum_log = 30

        multi_path_eval_kern[threads_per_block, blocks]( \
                        self.info_field, \
                        self.x_ticks, \
                        self.y_ticks, \
                        self.chan, \
                        paths_cuda, \
                        lengths, \
                        self.multi_reward, \
                        self.x_scale, \
                        self.y_scale, \
                        self.radius, \
                        RADIUS_DIST)


        num_blocks = lengths.shape[0] * self.info_field.shape[2]

        test_reward = self.multi_reward.copy_to_host()
        #print(test_reward[0,5:15,5:15,0])
        sum_rewards_kern[num_threads, num_blocks]( \
                            self.multi_reward, \
                            scores, \
                            reward_sum_log, \
                            self.info_field.shape[2], \
                            sum_per_thread)

        return scores * self.scales[np.newaxis, :]

    ## @override
    # getScore, gets the score of path given within the budget
    # @param path - the path as 2d numpy array (n x 2)
    # @param budget - the budget of the path, (typically path length)
    def getScore(self, path, budget=float('inf')):
        # force path along budget (TODO)
        threads_per_block = 32
        blocks = math.ceil(self.info_field.shape[0]*self.info_field.shape[1] / threads_per_block)

        #reward_arr = np.empty((self.info_field.shape[0], self.info_field.shape[1], len(self.chan)))
        #reward_arr = np.empty((len(self.chan), self.info_field.shape[0]*self.info_field.shape[1]))
        #reward_arr = np.ascontiguousarray(reward_arr)
        path_cuda = cuda.to_device(path)

        distance_from_path_kern[threads_per_block, blocks]( \
                        self.info_field, \
                        self.x_ticks, \
                        self.y_ticks, \
                        self.chan, \
                        path_cuda, \
                        self.reward_arr, \
                        self.x_scale, \
                        self.y_scale, \
                        self.radius, \
                        RADIUS_DIST)



        score = np.empty(len(self.chan))
        #reward_arr_shaped = np.empty((self.info_field.shape[0], self.info_field.shape[1], len(self.chan)))
        for i in range(len(self.chan)):
            # reward_arr_shaped[:,:,i] = np.reshape(reward_arr[i,:], \
            #         (self.info_field.shape[0], self.info_field.shape[1]),\
            #         order='C') # C, F, A
            score[i] = sum_reduce(self.reward_arr[0, :])

        return score * self.scales





## This function finds the distance of a point from a line segment.
#Based off of javascript function on stack overflow:
# https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
# @param x - the point x
# @param y - the point y
# @param x1 - x point of the line segment pt 1
# @param y1 - point of the line segment pt 1
# @param x2 - point of the line segment pt 2
# @param y2 - point of the line segment pt 2
@cuda.jit(device=True)
def distance_from_line_seg(x, y, x1, y1, x2, y2):
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A*C + B*D
    len_sq = C*C + D*D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    # find the closest point on line segment (xx, yy)
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param*C
        yy = y1 + param*D

    # find the distance from closest point
    dx = x - xx
    dy = y - yy
    return math.sqrt(dx*dx + dy*dy)

@cuda.jit(device=True)
def radius_dist(d, radius):
    if d <= radius:
        return 1
    else:
        return 0



## cuda kernel for finding the info field cost
# @param info_field - information field input (numpy field)
# @param x_ticks - the x ticks
# @param y_ticks - the y ticks
# @param path - numpy path
# @param reward_arr - information for the array.
#
# @return the reward vector
@cuda.jit
def distance_from_path_kern(info_field, \
                            x_ticks, \
                            y_ticks, \
                            chans, \
                            path, \
                            reward_arr, \
                            x_scale, \
                            y_scale, \
                            radius, \
                            func_to_call):
    # thread id
    tx = cuda.threadIdx.x
    # block id in 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # The flattened index
    pos = tx + ty * bw
    if pos < info_field.shape[0]*info_field.shape[1]:
        x_idx = int(pos / info_field.shape[1])
        y_idx = pos % info_field.shape[1]

        x = x_ticks[x_idx] + (x_scale / 3)
        y = y_ticks[y_idx] + (y_scale / 3)

        min_dist = 999999999999
        # find the minimum distance to the path
        for i in range(1, path.shape[0]):
            dist = distance_from_line_seg(x,y,path[i,0], path[i,1], path[i-1,0], path[i-1,1])
            if dist < min_dist:
                min_dist = dist

        # find the multiplier for function
        if func_to_call == RADIUS_DIST:
            multiplier = radius_dist(min_dist, radius)
        else:
            multiplier = 0

        for i, chan in enumerate(chans):
            #reward_arr[x_idx, y_idx, i] = multiplier #multiplier * info_field[x_idx, y_idx, chan]
            reward_arr[i, pos] = multiplier




@cuda.reduce
def sum_reduce(a,b):
    return a + b




@cuda.jit
def sum_rewards_kern(   reward_arr, \
                        output_reward, \
                        size_to_call_for_sum, \
                        num_chans,\
                        num_to_sum):
    # thread id
    path_pos = cuda.threadIdx.x
    # block id in 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # The flattened reward index

    if ty < num_chans * output_reward.shape[0]:

        k = ty % num_chans
        p_i = int(ty / num_chans)

        num_rewards = reward_arr.shape[1] * reward_arr.shape[2]


        ###### SUM the reward arrays
        for i in range(size_to_call_for_sum):
            pos_diff = num_to_sum**(i)
            #sum_idx = path_pos % pos_diff

            if path_pos*pos_diff < num_rewards:
                sum_idx = path_pos * pos_diff
                outer_x = int(sum_idx / reward_arr.shape[2])
                outer_y = sum_idx % reward_arr.shape[2]


                if sum_idx < num_rewards:
                    for j in range(1, num_to_sum):
                        sum_idx_sum = sum_idx + (j*pos_diff)
                        if sum_idx_sum < num_rewards:
                            inner_x = int(sum_idx_sum / reward_arr.shape[2])
                            inner_y = sum_idx_sum % reward_arr.shape[2]

                            reward_arr[p_i, outer_x, outer_y, k] += reward_arr[p_i, inner_x, inner_y, k]

            # sync threads for summing
            cuda.syncthreads()

        if path_pos == 0:
            output_reward[p_i, k] = reward_arr[p_i, 0, 0, k]




## cuda kernel for finding the info field cost
# @param info_field - information field input (numpy field)
# @param x_ticks - the x ticks
# @param y_ticks - the y ticks
# @param path - numpy path (num_paths, n, 2)
# @param reward_arr - information for the array.
#
# @return the reward vector
@cuda.jit
def multi_path_eval_kern(   info_field, \
                            x_ticks, \
                            y_ticks, \
                            chans, \
                            paths, \
                            lengths,\
                            reward_arr, \
                            x_scale, \
                            y_scale, \
                            radius, \
                            func_to_call):
    # thread id
    tx = cuda.threadIdx.x
    # block id in 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # The flattened index
    pos = tx + ty * bw
    if pos < info_field.shape[0]*info_field.shape[1]*lengths.shape[0]:
        num_rewards = info_field.shape[0]*info_field.shape[1]
        p_i = int(pos / num_rewards)
        path_pos = pos % num_rewards
        x_idx = int(path_pos / info_field.shape[1])
        y_idx = path_pos % info_field.shape[1]

        x = x_ticks[x_idx] + (x_scale / 3)
        y = y_ticks[y_idx] + (y_scale / 3)

        min_dist = 999999999999
        # find the minimum distance to the path
        for i in range(1, lengths[p_i]):
            dist = distance_from_line_seg(x,y,paths[p_i,i,0], paths[p_i,i,1], paths[p_i,i-1,0], paths[p_i,i-1,1])
            if dist < min_dist:
                min_dist = dist

        # find the multiplier for function
        if func_to_call == RADIUS_DIST:
            multiplier = radius_dist(min_dist, radius)
        else:
            multiplier = 0

        for i, chan in enumerate(chans):
            reward_arr[p_i, x_idx, y_idx, i] = multiplier * info_field[x_idx, y_idx, chan]
            #reward_arr[p_i,i, pos] = multiplier












# if __name__ == '__main__':
#     # a set of test functions
#
#     d = distance_from_line_seg(5,5,0,0, 10,2)
#     print(d)
#     d = distance_from_line_seg(5,5,0,2, 10,2)
#     print(d)
#     d = distance_from_line_seg(-12,5,0,0, 10,2)
#     print(d)
#     d = distance_from_line_seg(21,5,0,0, 10,2)
#     print(d)

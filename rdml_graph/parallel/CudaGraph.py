# CudaGraph.py
# Written Ian Rankin - May 2021
#
#

from numba import jit
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import math

from rdml_graph.parallel import get_connection_graph

def cuda_rand_traversal(G, start_N, budget):
    A, costs, num_edges = get_connection_graph(G)
    start_ids = np.array([n.id for n in start_N])

    threads_per_block = 64
    blocks = len(start_N) % threads_per_block
    rng_states = cuda.random.create_xoroshiro128p_states( \
            threads_per_block * blocks, seed=1)

    budgets = np.ones(len(start_N)) * budget
    max_path_depth = 200
    paths = np.ones((len(start_N), max_path_depth)) * -1

    kern_rand_travesal[blocks, threads_per_block](A, costs, num_edges, paths, start_ids, budgets, rng_states)


    return paths



@cuda.jit
def kern_rand_travesal(A, costs, num_edges, paths, start_locs, budgets, \
        rng_states):
    n = cuda.grid(1)

    if n < A.shape[0]:
        # traverse
        length = 0
        cur_node = start_locs[n]

        for i in range(paths.shape[1]):
            paths[n, i] = cur_node

            if num_edges[cur_node] == 0:
                break
            rand_num = dev_rand_int(rng_states, n, 0, num_edges[cur_node])

            length += costs[cur_node, rand_num]
            cur_node = A[cur_node, rand_num]

            if length >= budgets[n]:
                break





# @param rng_states
# @param n - the number of the index
# @param l_bound - lower bound
# @param u_bound - upper bound
#
# @return -
@cuda.jit(device=True)
def dev_rand_int(rng_states, n, l_bound, u_bound):
    r = cuda.random.xoroshiro128p_uniform_float32(rng_states, n)

    diff = u_bound - l_bound

    r = math.floor(r * diff)
    return int(r)

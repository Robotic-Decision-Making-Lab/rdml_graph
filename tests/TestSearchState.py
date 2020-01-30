# TestHSign.py
# Written Ian Rankin - January 2020
#
#



import rdml_graph as gr
import heapq
import numpy as np



z = 42

x = gr.SearchState(state=z, rCost=4)

print(x)

heap = []

for i in range(15):
    r = np.random.random()
    s = gr.SearchState(state=i,rCost=r, parent=x)
    s.invertCmp = True
    heapq.heappush(heap, s)

print('Top of Heap:')
topHeap = heapq.heappop(heap)
print(topHeap)

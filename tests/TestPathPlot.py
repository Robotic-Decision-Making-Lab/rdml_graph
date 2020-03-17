import numpy as np
import rdml_graph as gr
import matplotlib.pyplot as plt

pts = np.array([[0,0],[5,5],[5,-5],[10,7]])

print(pts)

gr.plot2DPath(pts)

plt.xlim(-10,10)
plt.ylim(-10,10)

plt.show()

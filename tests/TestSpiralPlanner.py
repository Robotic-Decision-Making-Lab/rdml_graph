## TestSpiralPlanner.py
# Written Ian Rankin July 2022
#
#


import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('test')
    map = {'x_ticks': np.arange(-10,11), 'y_ticks': np.arange(-10, 11)}
    spacing = 2
    start = np.array([-5.5, -3.6])

    plan = gr.SpiralPlan(start, map, spacing, True)

    gr.plot2DPath(plan)
    print(plan)
    plt.show()


if __name__ == "__main__":
    main()

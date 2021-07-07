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
## @package MultiObjectivePlot.py
# Written Ian Rankin - April 2021
#
# A set of code to quickly plot multi-objective information fields and paths

import numpy as np
import matplotlib.pyplot as plt

from rdml_graph.plot import plot2DPath

## plot_multi
# plots multiple objectives and paths
# @param info_field - the information field numpy(width, height, channels)
# @param paths - [opt] list of 2d numpy paths
# @param info_names - [opt] list of names for information fields
# @param path_names - [opt] list of names for the paths
# @param fig - [opt] the input figure
def plot_multi(info_field, paths=[], info_names=None, path_names=None, \
        cmap='viridis', radius=None, legend=True, fig= None):
    if fig is None:
        fig =plt.figure(constrained_layout=True)


    num_info = info_field.shape[2]
    height = 5

    colors = ['black', 'white', 'red', 'green', 'blue', 'purple', 'turquoise', \
            'crimson', 'navy', 'brown', 'yellow', 'orange', 'gray', 'indigo', \
            'lime', 'cyan', 'orangered', 'teal', 'magenta', 'peru', 'olive', \
            'aquamarine', 'orchid', 'lightcoral', 'khaki', 'dodgerblue', \
            'fuchsia', 'silver', 'navajowhite', 'limegreen']
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']

    if info_names is None:
        info_names = ['Objective '+str(i+1) for i in range(num_info)]

    if path_names is None:
        path_names = ['Path '+str(i+1) for i in range(len(paths))]

    # Handle different number of paths
    if num_info <= 4:
        gs = fig.add_gridspec(ncols=num_info+1, nrows=1, figure=fig, \
                width_ratios=([7]*num_info)+[2])
        fig.set_size_inches(height*num_info, height+0.5)



        for i in range(num_info):
            ax = fig.add_subplot(gs[0,i])
            ax.set_title(info_names[i])
            if i == 0:
                plt.ylabel('y (km)')
            plt.xlabel('x (km)')
            ax.imshow(info_field[:,:,i].transpose(), \
                        vmin=np.min(info_field[:,:,i]), \
                        vmax=np.max(info_field[:,:,i]), \
                        origin='lower', cmap=cmap)
            ps = []

            for j, path in enumerate(paths):
                p = plot2DPath(path, color=colors[j % len(colors)], \
                                label=path_names[j], radius=radius, \
                                line_style=line_styles[j % len(line_styles)])
                ps.append(p)
            # end for (paths)
        # end for (info field)
        ax = fig.add_subplot(gs[0,num_info])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        if legend:
            plt.legend(handles=ps, loc='center')
    else:
        raise NotImplemented('Num info larger than 4 is not implemented currently')
    # end if num_info

    return fig, ps

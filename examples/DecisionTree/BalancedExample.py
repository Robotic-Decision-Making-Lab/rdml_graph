# BalancedExample.py
# Written Ian Rankin - September 2022
#
# An example of a balanced tree (kd-tree)

import rdml_graph as gr

import pdb

if __name__ == "__main__":
    import sklearn.datasets as dt

    iris = dt.load_iris()
    X = iris.data
    y = iris.target

    root, _ = gr.create_balanced_decision_tree(X)
    view = root.get_viz(labels=True)
    view.view()

    print(X[10])
    ans = root.traverse(X[10])

    print(ans)

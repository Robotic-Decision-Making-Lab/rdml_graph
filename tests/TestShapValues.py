
import rdml_graph as gr
import sklearn.datasets as dt
import matplotlib.pyplot as plt
import numpy as np
import random


#################################### Test functions #######################




def main():
    test_list = [0.01]
    print(gr.select_SHAP_idx(test_list, 3))

    num_samps = 100
    num_dim = 2

    xs = [[random.uniform(0,10) for j in range(num_dim)]  for i in range(num_samps)]
    X = [(x, f(x)+random.uniform(-0.5, 0.5)) for x in xs]

    #print(xs)
    #print(X)

    types = ['float'] * num_dim

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100)

    t = root.get_viz(labels=True)
    t.view()

    x = [5,5]

    prediction = root.traverse(x)
    shap = gr.TreeSHAP(x, root)

    print('TreeSHAP_int')
    shap_int = gr.TreeSHAP_INT(x, root)

    print(prediction)
    print(shap)
    print(shap_int)
    print(sum(shap))


if __name__ == '__main__':

    def f(x_in):
        y_0 = np.sin(x_in[0]-.3572)*4-0.2
        y_1 = (np.cos(x_in[1]*1.43)-.3572)*3
        #y_2 = x_in[2]

        return y_0 + y_1# + y_2

    main()

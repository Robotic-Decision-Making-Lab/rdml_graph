
import rdml_graph as gr
import numpy as np


if __name__ == '__main__':
    # test code
    shap_values = np.random.random((16, 5))

    alts_to_show, features = gr.select_alts_from_shap_diff(0, shap_values, 3, 'similar_except')
    print('Alts to show: ' + str(alts_to_show)+' pertinent_features: '+str(features))
    print(shap_values[alts_to_show[0]])
    print(shap_values[alts_to_show[1]])
    print(shap_values[alts_to_show[2]])

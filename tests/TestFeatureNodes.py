# TestFeatureNodes.py
# Written Ian Rankin March 2020
#
#

import rdml_graph as gr



feat1 = gr.FeatureNode(0,"Shaw Island", keywords=['Island', 'Shaw', 'Isle'])
feat2 = gr.FeatureNode(1, "uf-1", keywords=['Upwelling Front', 'Front', 'Coastal front', 'Coastal upwelling front'])
feat3 = gr.FeatureNode(2, "hotspot1")


print(feat1)
print(feat2)
print(feat3)

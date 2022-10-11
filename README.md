# rdml_graph
[![Python Testing](https://github.com/Robotic-Decision-Making-Lab/rdml_graph/actions/workflows/python-test.yml/badge.svg)](https://github.com/Robotic-Decision-Making-Lab/rdml_graph/actions/workflows/python-test.yml)


Set of graph function including:
graph
A*
Homotopic augmented graphs



## Dependencies

numpy>=1.14.0

matplotlib>=2.1.2


## Installation
From the root directory of rdml_graph:

```
pip install -e . --user
```

For full features also run
```
sudo apt-get install doxygen graphviz
```

## Documentation

Doxygen documentation is used for the library. This allows creation of an easy
to use html files. To generate the documentation...
```
sudo apt-get install doxygen
cd doc
doxygen Doxyfile
```


## Usage

```
import rdml_graph as gr

hSign = gr.HomologySignature(5)
```

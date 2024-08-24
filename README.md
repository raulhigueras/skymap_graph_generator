# SkyMap

<p align="center">
<img alt="Python versions: 9.9 9.10 9.11 9.12" src="https://img.shields.io/badge/Python-3.9_3.10_3.11_3.12-green">
<a href="https://github.com/raulhigueras/skymap_graph_generator/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
<img alt="inport style: isort" src="https://img.shields.io/badge/import_style-isort-blue">
<img alt="code style: black" src="https://img.shields.io/badge/code_style-black-blue">
</p>

SkyMap is a generative graph model designed for GNN benchmarking with two main usecases:
1. Generate graphs with a similar GNN behaviour to an input graph.
2. Generate diverse synthetic graph datasets for benchmarking.

This document explains how to install and the basic usage of SkyMap. For a more comprehensive and detailed explanation of the features, please check the [examples](examples)

## Installation

To install skymap using [pip](https://pypi.org):

```{bash}
git clone https://github.com/raulhigueras/skymap_graph_generator
cd skymap_graph_generator
pip install .
```

To install skymap using [poetry](https://python-poetry.org):

```{bash}
git clone https://github.com/raulhigueras/skymap_graph_generator
cd skymap_graph_generator
poetry install
```

## Graph Generation

SkyMap can be used to imitate graphs. This is, to generate graphs with similar characteristics to an input graph. 

> :warning: When using arbitrary input graphs, make sure that the graphs has the following format:
> - Each node must have a property `y` with an integer representing the class.
> - Each node must have a property `x` with the feature vector in form of a numerical list.
> - All features must be binary.  

### Command-line Interface (CLI)

SkyNet offers directly a CLI to use the generator. The input graph must be in [.gml](https://en.wikipedia.org/wiki/Graph_Modelling_Language) format.

```{bash}
skynet mimic-graph -g <path to input graph (.gml)> -o <output directory> -n <number of nodes (optional)>
```

Note: if no number of nodes is provided, the number of nodes of the input graph will be used.

### Python usage

Another way of generating graphs is directly using the project as a library:

```{python}
from sjymap import SkyMap
import networkx as nx

skymap = SkyMap()
input_graph: nx.Graph = (...) # <- your graph here as a networkx Graph
gen_graph: nx.Graph = skymap.mimic_graph(input_graph, num_nodes=1000)
```

A more comprehensive guide of how to use the python library to generate graphs can be found in [Examples](examples/use_skymap.ipynb)

<!--
## Graph Dataset Generation

SkyMap can also be used to generate diverse and complete synthetic graph datasets. Similarly to the graph generation, this feature can be used through a CLI or directly with the Python library.

To generate the dataset, a _Dataset Specification file_ must be provided. This file is a TOML file defining the minimum and maximum value to take for each parameter of the model, and the distribution to use. A small example:

```{toml}
[num_nodes]
start=1000         # Only graphs with 1000 nodes
end=1000

[num_classes]
start=3            # Graphs with 3 to 5 classes, uniformly
end=5

[density]
start=0          # Graphs with densities between 10^0/1000 and 10^3/1000
end=3
distribution="10^/1000"
```

The possible values of distributions are: `uniform` (default), `2^`, `10^`, `10^/1000`. A full example can be seen in [Example Specification File](examples/data/dataset_specs_example.toml)

### Command Line Interface (CLI)

The CLI command for the dataset generation looks like this:

```{bash}
skynet generate-dataset -p <path to dump graphs> -s <dataset specification file> -n <number of graphs>
```

### Python Library

Another way of generating a dataset is through the Python library:

```{python}
output_path = "path/to/dump/dataset"
num_graphs = 3

generator = DatasetGenerator(path=output_path)
generator.generate(specs, num_graphs=num_graphs)
```

A more comprehensive guide of how to use the python library to generate a synthetic graph dataset can be found in [Examples](examples/generate_dataset.ipynb)

-->

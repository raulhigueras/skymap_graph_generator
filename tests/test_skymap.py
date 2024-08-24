from skymap import SkyMap
from skymap.metrics import SkyMapMetrics
import pytest
import networkx as nx


def test_SkyMap_init():
    skymap = SkyMap()
    assert isinstance(skymap, SkyMap)
    
    
def test_SkyMap_mimic_graph_1(graph_1):
    skymap = SkyMap()
    graph_mimic = skymap.mimic_graph(graph_1)
    assert isinstance(graph_mimic, nx.Graph)
    assert nx.number_of_nodes(graph_1)*2 == nx.number_of_nodes(graph_mimic)
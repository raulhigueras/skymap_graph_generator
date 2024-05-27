from skynet import SkyNet
from skynet.metrics import SkyNetMetrics
import pytest
import networkx as nx


def test_skynet_init():
    skynet = SkyNet()
    assert isinstance(skynet, SkyNet)
    
    
def test_skynet_mimic_graph_1(graph_1):
    skynet = SkyNet()
    graph_mimic = skynet.mimic_graph(graph_1)
    assert isinstance(graph_mimic, nx.Graph)
    assert nx.number_of_nodes(graph_1)*2 == nx.number_of_nodes(graph_mimic)
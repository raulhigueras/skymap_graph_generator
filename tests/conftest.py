import networkx as nx
import pytest
from skymap.metrics import SkyMapMetrics

@pytest.fixture
def graph_1():
    g = nx.wheel_graph(5)
    nodes = [0, 1, 2, 3, 4]
    classes = [0, 0, 1, 1, 1]
    features = [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ]
    nx.set_node_attributes(g, dict(zip(nodes, classes)), "y")
    nx.set_node_attributes(g, dict(zip(nodes, features)), "x")

    return g

@pytest.fixture()
def metrics_1():
    return SkyMapMetrics(
        num_nodes=100,
        density=0.2,
        class_imbalance_ratio=0.5,
        log_logistic_delta=0.1,
        log_logistic_lambda=0.2,
        dist_x_mean=0.2,
        dist_y_mean=0.2,
        dist_affinity_mean=0.3,
        self_affinity_imbalance_ratio=0.4,
        interclass_affinity_imbalance_ratio=0.1,
        homophily=0.7,
        num_features=20,
        zero_gen_means_mean=0.5,
        zero_gen_means_var=0.3,
        zero_gen_vars_mean=0.2,
        zero_gen_vars_var=0.4
    )
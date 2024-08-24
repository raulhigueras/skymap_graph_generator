import networkx as nx
import pytest

from skymap.metrics import *


def float_equal(num1: float, num2: float, precision: float = 1e-3):
    return abs(num1 - num2) < precision



def test_basic_metrics_1(graph_1):
    metrics = compute_basic_metrics(graph_1)

    assert metrics["num_nodes"] == 5
    assert metrics["density"] == 0.8
    assert metrics["mean_degree"] == 3.2
    assert metrics["max_degree"] == 4
    assert float_equal(metrics["clustering"], 0.666)
    assert float_equal(metrics["degree_assortativity"], -0.333)
    assert metrics["largest_component_perc"] == 1


def test_degree_distribution_metrics_1(graph_1):
    metrics = compute_degree_distribution_metrics(graph_1)

    assert float_equal(metrics["log_logistic_delta"], 100.133)
    assert float_equal(metrics["log_logistic_lambda"], 4.931)


def test_laplacian_matrix_metrics_1(graph_1):
    metrics = compute_laplacian_matrix_metrics(graph_1)

    assert float_equal(metrics["eigen1"], 0)

def test_class_distribution_metrics_1(graph_1):
    metrics = compute_class_distribution_metrics(graph_1)

    assert metrics["num_classes"] == 2
    assert float_equal(metrics["class_imbalance_ratio"], 0.2)

def test_feature_distribution_metrics_1(graph_1):
    metrics = compute_feature_distribution_metrics(graph_1)

    assert metrics["num_features"] == 4
    assert metrics["perc_0"] == 0.65
    assert "zero_gen_means_mean" in metrics
    assert "zero_gen_means_var" in metrics
    assert "zero_gen_vars_mean" in metrics
    assert "zero_gen_vars_var" in metrics
    assert "perc_nodes_conc_mean" in metrics
    assert "perc_nodes_conc_var" in metrics
    assert "perc_edges_conc_mean" in metrics
    assert "perc_edges_conc_var" in metrics
    assert "deg_mult_mean" in metrics
    assert "deg_mult_var" in metrics
    assert "corr_mean" in metrics
    assert "corr_var" in metrics

def test_mixing_matrix_metrics(graph_1):
    metrics = compute_mixing_matrix_metrics(graph_1)

    assert metrics["homophily"] == 0.375
    assert "self_affinity_imbalance_ratio" in metrics
    assert "interclass_affinity_imbalance_ratio" in metrics
    assert "dist_affinity_mean" in metrics
    assert "dist_affinity_std" in metrics
    assert "dist_x_mean" in metrics
    assert "dist_y_mean" in metrics

def test_skymap_metrics_from_graph(graph_1):
    metrics = SkyMapMetrics.from_graph(graph_1)
    assert isinstance(metrics, SkyMapMetrics)
    assert metrics.num_nodes == 5

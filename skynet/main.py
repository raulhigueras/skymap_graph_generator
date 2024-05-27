import networkx as nx

from skynet.metrics import SkyNetMetrics
from typing import Literal
from skynet.utils import imbalance_distribution, distance, pmf_discrete_log_logistic, gen_beta_moments, pmf_disc_beta
import numpy as np
import random
import math
import itertools as it
import functools as ft
import pandas as pd


class SkyNet:

    def __init__(
        self, 
        subgraphs_gen_method: Literal["search", "random"] = "search"
    ):
        self.subgraphs_gen_method = subgraphs_gen_method

    def mimic_graph(self, g: nx.Graph, num_nodes: int = None) -> nx.Graph:
        metrics = SkyNetMetrics.from_graph(g)
        if num_nodes is not None:
            metrics.num_nodes = num_nodes
        return self.generate_graph(metrics)
    
    def generate_random_subgraphs(self, metrics: SkyNetMetrics, mixing_matrix: np.ndarray) -> list[nx.Graph]:
        
        nodes_per_class = imbalance_distribution(metrics.num_classes, metrics.class_imbalance_ratio)
        nodes_per_class = nodes_per_class[::-1]
        classes = range(metrics.num_classes)
        
        subgraphs: list[nx.Graph] = []
        for c in classes:
            num_nodes = math.ceil(nodes_per_class[c] * metrics.num_nodes)
            num_edges = math.ceil(metrics.num_edges * mixing_matrix[c,c])
            subgraph = nx.gnm_random_graph(num_nodes, num_edges)
            nx.set_node_attributes(subgraph, c, name="y")
            subgraphs.append(subgraph)
            
        return subgraphs
    
    def search_subgraphs(self, metrics: SkyNetMetrics, mixing_matrix: np.ndarray) -> list[nx.Graph]: 
        
        nodes_per_class = imbalance_distribution(metrics.num_classes, metrics.class_imbalance_ratio)
        nodes_per_class = nodes_per_class[::-1]
        classes = range(metrics.num_classes)
        
        subgraphs: list[nx.Graph] = []
        for c in classes:
            num_nodes = math.ceil(nodes_per_class[c] * metrics.num_nodes)
            num_edges = math.ceil(metrics.num_edges * mixing_matrix[c,c])
            
            log_logistic = ft.partial(pmf_discrete_log_logistic, d=metrics.log_logistic_delta, l=metrics.log_logistic_lambda)
            ps = [log_logistic(i) for i in range(0,metrics.num_nodes)]
            exp_degrees = np.array(random.choices(range(0, metrics.num_nodes), weights=ps, k=metrics.num_nodes), dtype=float)
            exp_degrees[exp_degrees == 0] = 1e-1
            subgraph = nx.Graph()
            subgraph.add_nodes_from(range(metrics.num_nodes))

            posible_edges = list(it.combinations(range(metrics.num_nodes), 2))
            num_edges = min(num_edges, len(posible_edges))
            w =  np.array([exp_degrees[n1] * exp_degrees[n2] for n1,n2 in posible_edges])
            w = w / w.sum()
            edges_to_add = np.random.choice(range(len(posible_edges)), size=num_edges, replace=False, p=w)
            subgraph.add_edges_from(np.array(posible_edges)[edges_to_add])
            
            nx.set_node_attributes(subgraph, c, name="y")
            subgraphs.append(subgraph)

        return subgraphs

    def reconstruct_mixing_matrix(
        self, 
        metrics: SkyNetMetrics, 
        make_symmetric: bool = True,
        max_num_iters: int = 10_000
    ) -> np.ndarray:   
             
        metrics_objective = [metrics.dist_x_mean, metrics.dist_y_mean, metrics.dist_affinity_mean]
        diag = imbalance_distribution(metrics.num_classes, metrics.self_affinity_imbalance_ratio) 
        diag = metrics.homophility * (diag / sum(diag))

        X = np.zeros((metrics.num_classes, metrics.num_classes), float)
        np.fill_diagonal(X, diag)
        idx=np.flip(np.argsort(np.diag(X)))
        X = X[idx,:][:,idx]

        other = imbalance_distribution((metrics.num_classes - 1) * metrics.num_classes / 2, metrics.interclass_affinity_imbalance_ratio) * (1 - metrics.homophility)
        dist  = math.inf
        for i in range(max_num_iters):

            random.shuffle(other)
            other_proposed = other.copy()

            dist_affinity_array = [0] * metrics.num_classes
            X_i = np.zeros((metrics.num_classes, metrics.num_classes), float)

            index = len(other_proposed) -1
            for i in range(metrics.num_classes):
                for j in reversed(range(i + 1, metrics.num_classes)):
                    X_i[i,j] = other_proposed[index]
                    index -= 1

            dist_x_list = [0] * metrics.num_classes 
            dist_y_list = [0] * (metrics.num_classes - 1)
            for c1 in range(metrics.num_classes):
                    for c2 in range(c1 + 1, metrics.num_classes):
                            dist_affinity_array[c2-c1] += X_i[c1,c2]
                            dist_x_list[c1] += X_i[c1,c2]
                            dist_y_list[c2-1] += X_i[c1,c2]

            dist_affinity_array = np.array(dist_affinity_array) / np.array(np.arange(metrics.num_classes,0, -1))
            dist_affinity_array = dist_affinity_array / sum(dist_affinity_array)
            dist_affinity_mean = 0
            for i in range(len(dist_affinity_array)):
                dist_affinity_mean += dist_affinity_array[i] * i

            dist_x_list = np.array(dist_x_list) / np.array(np.arange(metrics.num_classes,0, -1))
            dist_x_list = dist_x_list / sum(dist_x_list) 
            dist_x_mean = 0
            for i in range(len(dist_x_list)):
                dist_x_mean += dist_x_list[i] * i

            dist_y_list = np.array(dist_y_list) / np.array(np.arange(1, metrics.num_classes))
            dist_y_list = dist_y_list / sum(dist_y_list) 
            dist_y_mean = 0
            for i in range(len(dist_y_list)):
                dist_y_mean += dist_y_list[i] * i
            metrics_proposed = [dist_x_mean, dist_y_mean, dist_affinity_mean]
            dist_proposed = distance(metrics_proposed, metrics_objective)
            if dist_proposed < dist:
                dist = dist_proposed
                other = other_proposed
            

        index = len(other) -1
        for i in range(metrics.num_classes):
            for j in reversed(range(i + 1, metrics.num_classes)):
                X[i,j] = other[index]
                index -= 1

        if make_symmetric:
            X = X + X.T - np.diag(np.diag(X))

        return X
        
    
    def assign_properties(
        self, 
        graph: nx.Graph, 
        metrics: SkyNetMetrics,
        rho_mean: float = None,
        rho_var: float = None
        ) -> nx.Graph:
        ...
        classes = range(metrics.num_classes)
        means = gen_beta_moments(metrics.zero_gen_means_mean, metrics.zero_gen_means_var, metrics.num_classes)
        stds = gen_beta_moments(metrics.zero_gen_vars_mean, metrics.zero_gen_vars_mean, metrics.num_classes)
        A_c = np.array([gen_beta_moments(e[0], e[1], metrics.num_features) for e in zip(means,stds)])

        # return assign_properties_matrixes(G, num_features, A_c, rho_by_class = rho_by_class)
    
        H = nx.Graph()
        H.add_nodes_from(sorted(graph.nodes(data=True)))
        H.add_edges_from(graph.edges(data=True))

        node_data = pd.DataFrame.from_dict([{"node": i[0], "y": i[1]['y'] } for i in  H.nodes.items()])
        # classes = list(set(node_data["y"]))
        num_classes = len(classes)  

        deg_data = pd.DataFrame.from_dict([{
            "node": d[0], 
            "k": d[1], 
            "neigh_by_class": [node_data[node_data['node'].isin(graph.neighbors(d[0]))]['y'].eq(c).sum() for c in classes]
            } for d in  H.degree()])
        

        deg_node_data = pd.merge(node_data, deg_data, on="node")
        deg_node_data["neigh_k_by_class"] = [[deg_node_data[deg_node_data['node'].isin([e for e in graph.neighbors(e["node"])] + [e["node"]]) & deg_node_data['y'].eq(c)]["k"].sum() for c in classes] for e in deg_node_data.iloc]
        deg_node_data["neigh_k_internal"] = [e["neigh_k_by_class"][e['y']] for e in deg_node_data.iloc]

        node_classes = np.array([i[1]['y'] for i in  H.nodes.items()])
        
        # A = gen_properties(A_c, classes, H.number_of_nodes(), metrics.num_features, None, None, rho_by_class) #waights)
        rho = None
        w = None
        if w is None:
            w = np.ones((metrics.num_features, nx.number_of_nodes(graph)))
        props = np.array([[1] * metrics.num_features] * nx.number_of_nodes(graph))
        for c in classes:
            idx_all = np.where(node_classes==c)[0]
            for f in range(metrics.num_features):
                p = A_c[c, f]
                class_w = w[f][idx_all]
                class_w /= sum(class_w)
                idx = np.random.choice(len(idx_all), size=int(len(idx_all)* p), replace=False, p=class_w)
                props[idx_all[idx], f] = 0
        A = np.array(props)

        for n in H.nodes:
            H.nodes[n]['x'] = A[int(n)].tolist()
        return H
    

    def add_inter_edges(self, graph: nx.Graph, metrics: SkyNetMetrics, mixing_matrix: np.ndarray) -> nx.Graph:
        node_data = pd.DataFrame.from_dict([{"node": i[0], "y": i[1]['y'] } for i in  graph.nodes.items()])
        deg_data = pd.DataFrame.from_dict([{"node": d[0], "k": d[1] } for d in  graph.degree()])
        deg_node_data = pd.merge(node_data, deg_data, on="node")
        classes = range(metrics.num_classes)
        for c1 in classes:
            for c2 in range(c1+1, metrics.num_classes):
                num_edges = math.ceil(metrics.num_edges * mixing_matrix[c1,c2])
                c1_data = deg_node_data[deg_node_data["y"] == classes[c1]]
                c2_data = deg_node_data[deg_node_data["y"] == classes[c2]]
                
                edges_to_add = set()
                perc_edges_conc = gen_beta_moments(metrics.perc_edges_conc_mean, metrics.perc_edges_conc_var, 2)
                perc_nodes_conc = gen_beta_moments(metrics.perc_nodes_conc_mean, metrics.perc_nodes_conc_var, 2)

                num_edges_conc_c1 = math.floor(num_edges * perc_edges_conc[0])
                num_edges_conc_c2 = math.floor(num_edges * perc_edges_conc[1])
                num_edges_deg = num_edges - num_edges_conc_c1 - num_edges_conc_c2
                edges_conc_1 = set(random.choices(list(c1_data["node"]), k = math.ceil(len(c1_data["node"]) * perc_nodes_conc[0])))
                edges_conc_2 = set(random.choices(list(c2_data["node"]), k = math.ceil(len(c2_data["node"]) * perc_nodes_conc[1])))
                
                posible_edges_conc1 = list(it.product(edges_conc_1, set(c2_data["node"])))
                edges_to_add.update(random.choices(posible_edges_conc1, k=num_edges_conc_c1))
                
                posible_edges_conc2 = list(set(it.product(edges_conc_2, set(c1_data["node"]))) - edges_to_add)
                edges_to_add.update(random.choices(posible_edges_conc2, k=num_edges_conc_c2))

                posible_edges = list(set(it.product(set(c1_data["node"]), set(c2_data["node"]))))

                deg_mult_mean = metrics.deg_mult_mean
                deg_mult_var = metrics.deg_mult_var

                c1_max_degree = np.array(c1_data["k"]).max() + 1
                c2_max_degree = np.array(c2_data["k"]).max() + 1
                
                c1_deg_counts = c1_data['k'].value_counts()
                c2_deg_counts = c2_data['k'].value_counts()

                if num_edges_deg > 0:
                    w =  np.array(pmf_disc_beta(
                                    np.array([np.sqrt(c1_k*c2_k/(c1_max_degree*c2_max_degree))
                                    for c1_k,c2_k in it.product(c1_data['k'], c2_data['k'])]), deg_mult_mean, deg_mult_var
                                )) / np.array([c1_deg_counts[c1_k]*c2_deg_counts[c2_k] for c1_k,c2_k in it.product(c1_data["k"], c2_data["k"])])
                    w = w / w.sum()
                    edges_to_add.update(random.choices(posible_edges, k=num_edges_deg, weights=w))
                graph.add_edges_from(edges_to_add)
        return graph


    def generate_graph(self, metrics: SkyNetMetrics) -> nx.Graph:
        
        # Reconstruct Mixing Matrix
        mixing_matrix = self.reconstruct_mixing_matrix(metrics)
        
        # Generate subgraphs
        match self.subgraphs_gen_method:
            case "search":
                subgraphs = self.search_subgraphs(metrics, mixing_matrix)
            case "random":
                subgraphs = self.generate_random_subgraphs(metrics, mixing_matrix)
            case _:
                raise AttributeError(f"Invalid {self.subgraphs_gen_method=}")        
        
        # Combine subgraphs
        G = nx.disjoint_union_all(subgraphs) 

        # Assign Properties
        G = self.assign_properties(G, metrics)
        
        # Add edges to combine subgraphs
        G = self.add_inter_edges(G, metrics, mixing_matrix)
        
        return G
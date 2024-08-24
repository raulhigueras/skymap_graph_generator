import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation.clustering_coefficient import (
    average_clustering as average_clustering_approx,
)
from networkx.algorithms.cluster import average_clustering
from scipy.sparse.linalg import eigsh
from skymap.utils import moments_joint_prob, imb_distr_fit, get_mixing_matrix, imb_distr_fit_likelihood

from skymap.utils import fit_degree_distribution

MAX_NUM_NODES_CLUSTERING_APPROX = 10_000
MAX_CLUSTER_COEFF_TRIALS = 1_000


def compute_basic_metrics(g: nx.Graph) -> dict[str, float]:
    degrees = [x[1] for x in nx.degree(g)]
    largest_component = g.subgraph(max(nx.connected_components(g), key=len))
    num_nodes = nx.number_of_nodes(g)
    if num_nodes > MAX_NUM_NODES_CLUSTERING_APPROX:
        clustering_coeff = average_clustering_approx(g)
    else:
        clustering_coeff = average_clustering(g)
    return dict(
        num_nodes=num_nodes,
        density=nx.density(g),
        mean_degree=np.mean(degrees),
        max_degree=np.max(degrees),
        max_degree_per=np.mean(degrees) / num_nodes,
        clustering=clustering_coeff,
        degree_assortativity=nx.degree_assortativity_coefficient(g),
        largest_component_perc=nx.number_of_nodes(largest_component) / num_nodes,
    )


def compute_degree_distribution_metrics(g: nx.Graph) -> dict[str, float]:
    delta, lamda = fit_degree_distribution(g)
    return dict(log_logistic_delta=delta, log_logistic_lambda=lamda)


def compute_laplacian_matrix_metrics(g: nx.Graph) -> dict[str, float]:
    largest_component = g.subgraph(max(nx.connected_components(g), key=len))
    assert len(largest_component) > 1  # Largest comp has at least one node
    laplacian = nx.normalized_laplacian_matrix(largest_component)
    eigenvals = eigsh(laplacian, return_eigenvectors=False, k=1, which="SM")
    return dict(eigen1=eigenvals[0])


def compute_class_distribution_metrics(g: nx.Graph) -> dict[str, float]:
    data_dict = [{"node": i[0], "y": i[1]["y"], "x": i[1]["x"]} for i in g.nodes.items()]
    node_data = pd.DataFrame.from_dict(data_dict)
    num_classes = len(node_data.y.unique())
    
    nodes_per_class = node_data['y'].value_counts().to_numpy()
    perc_nodes = nodes_per_class / nodes_per_class.sum()
    k, class_imbalance_ratio, class_imbalance_ratio_tendency  = imb_distr_fit(perc_nodes)
    
    return dict(
        num_classes=num_classes,
        class_imbalance_ratio=class_imbalance_ratio,
    )


def compute_feature_distribution_metrics(g: nx.Graph) -> dict[str, float]:
    
    data_dict = [{"node": i[0], "y": i[1]["y"], "x": i[1]["x"]} for i in g.nodes.items()]
    node_data = pd.DataFrame.from_dict(data_dict)
    data_dict = [{"node": d[0], "k": d[1], "k_internal": node_data[node_data['node'].isin(g.neighbors(d[0]))]['y'].eq(node_data[node_data['node'].eq(d[0])]['y'].iloc[0]).sum() } for d in g.degree()]
    deg_data = pd.DataFrame.from_dict(data_dict)
    deg_node_data = pd.merge(node_data, deg_data, on="node")
    deg_node_data['x_per'] = deg_node_data.apply(lambda x: sum(x['x']) / len(x['x']), axis=1)

    num_classes = len(node_data.y.unique())
    num_features = len(node_data.x.loc[0])
    
    # TODO: Refactor
    class_feature_data = node_data
    class_feature_data['idx'] = class_feature_data.index
    class_feature_data = class_feature_data.explode('x', ignore_index=True)
    class_feature_data = class_feature_data.rename(columns={"x": "value", "y": "class"})
    class_feature_data['feature'] = class_feature_data.groupby('idx').cumcount()

    class_feature_data_norm = class_feature_data

    class_feature_data_norm["value"] = \
        class_feature_data_norm["value"] / \
        class_feature_data_norm[abs(class_feature_data_norm["value"]) > sys.float_info.epsilon]["value"].abs().mean()

    feature_vec = class_feature_data_norm["value"]
    zero_mask = np.array(abs(feature_vec) < sys.float_info.epsilon)
    non_cero_features = feature_vec[~zero_mask]
    feature_vec_abs = np.abs(non_cero_features)

    num_features = len(node_data["x"].iloc[0])
    perc_0 = sum(zero_mask) / len(feature_vec)

    class_feature_matrix_zeros = np.zeros((num_classes, num_features))
    classes = list(set(node_data["y"]))
    for c in classes:
        feature_vec_c = class_feature_data_norm[(class_feature_data_norm["class"] == c)]
        c_i = classes.index(c)
        means = feature_vec_c.groupby("feature")["value"].mean()
        for k,v in means.items():
            class_feature_matrix_zeros[c_i, k] = 1 - v


    zero_gen_means = np.array([row.mean() for row in class_feature_matrix_zeros])
    zero_gen_vars = np.array([row.var()  for row in class_feature_matrix_zeros])
    zero_gen_means_mean = zero_gen_means.mean()
    zero_gen_means_var = zero_gen_means.var()
    zero_gen_vars_mean = zero_gen_vars.mean()
    zero_gen_vars_var = zero_gen_vars.var()
    
    
    list_intdeg = []
    list_num_edges_concentrated_links = []
    list_perc_edges_concentrated_links = []
    mixing_matrix_abs = np.zeros((num_classes, num_classes))
    for c1 in classes:
        neigbour_classes = []
        nodes = deg_node_data[deg_node_data["y"] == c1]        
        neigh_ids = []
        neigh_deg_internal = []
        deg_x_internal = []
        for _,d in nodes.iterrows():
            neighs = list(g.neighbors(d['node']))# + [d['node']]
            neighbours = deg_node_data[deg_node_data['node'].isin(neighs)]
            neigh_classes = np.array(neighbours['y'])            
            neigh_ids.extend(np.array(neighbours['node']))
            neigh_deg_internal.extend(np.array(neighbours['k_internal']))
            deg_x_internal.extend([d['k_internal']] * len(np.array(neighbours['k_internal'])))
            if len(neigh_classes) > 0:
                neigbour_classes.extend(neigh_classes)
        for c2 in classes:
            mixing_matrix_abs[classes.index(c1)][classes.index(c2)] = neigbour_classes.count(c2)
            if c1!=c2:
                raw_interk = np.array(neigh_ids)[np.array(neigbour_classes) == c2]
                data_inerk = np.unique(raw_interk, return_counts=True)
                indexes = data_inerk[1] > data_inerk[1].mean() + 3 * data_inerk[1].std()
                x = np.array(deg_x_internal)[np.array(neigbour_classes) == c2]/ (deg_node_data[deg_node_data["y"]==c1]["k_internal"].max() + 1)
                y = np.array(neigh_deg_internal)[np.array(neigbour_classes) == c2] / (deg_node_data[deg_node_data["y"]==c2]["k_internal"].max() + 1)
                list_intdeg.extend(np.sqrt(x*y))
                if len(set(data_inerk[0])) > 0:
                    list_num_edges_concentrated_links.append((indexes).sum() / len(set(data_inerk[0])))
                if sum(data_inerk[1]) > 0:
                    list_perc_edges_concentrated_links.append(sum(data_inerk[1][indexes]) /sum(data_inerk[1]))

    for c in range(num_classes):
        mixing_matrix_abs[c, c] = mixing_matrix_abs[c,c] /2
        
    corrs = []
    for c in classes:
        class_nodes = node_data[node_data['y'].eq(c)]
        assert len(class_nodes) > 1  # At least 2 nodes per class
        _, probs = moments_joint_prob(class_nodes)
        #np.savetxt(Path(out_path ,f"cov_{dataset.lower()}_{c}.txt"), cov)
        corrs.extend(probs)

    
    return dict(
        num_features=num_features,
        perc_0=perc_0,
        zero_gen_means_mean=zero_gen_means_mean,
        zero_gen_means_var=zero_gen_means_var,
        zero_gen_vars_mean=zero_gen_vars_mean,
        zero_gen_vars_var=zero_gen_vars_var,
        perc_nodes_conc_mean=np.mean(list_num_edges_concentrated_links), 
        perc_nodes_conc_var=np.var(list_num_edges_concentrated_links),
        perc_edges_conc_mean=np.mean(list_perc_edges_concentrated_links), 
        perc_edges_conc_var=np.var(list_perc_edges_concentrated_links),
        deg_mult_mean=np.mean(list_intdeg), 
        deg_mult_var=np.var(list_intdeg),
        corr_mean=np.mean(corrs),
        corr_var=np.var(corrs)
    )


def compute_mixing_matrix_metrics(g: nx.Graph) -> dict[str, float]:
    mixing_matrix = get_mixing_matrix(g)
    
    num_classes = len(mixing_matrix)

    self_affinity_list = []
    interclass_affinity_list = []
    dist_affinity = [0] * num_classes

    for c1 in range(num_classes):
            self_affinity_list.append(mixing_matrix[c1,c1])
            for c2 in range(c1 + 1, num_classes):
                    interclass_affinity_list.append(mixing_matrix[c1,c2])
                    dist_affinity[c2-c1] += mixing_matrix[c1,c2]

    self_affinity_list = np.array(self_affinity_list)
    interclass_affinity_list = np.array(interclass_affinity_list)

    homophily = sum(self_affinity_list)
    self_affinity_list_norm = self_affinity_list / sum(self_affinity_list)
    self_affinity_imbalance_ratio = imb_distr_fit_likelihood(self_affinity_list_norm)
    interclass_affinity_list_norm = interclass_affinity_list / sum(interclass_affinity_list)
    interclass_affinity_imbalance_ratio = imb_distr_fit_likelihood(interclass_affinity_list_norm)

    dist_affinity_array = [0] * num_classes
    dist_x_list = [0] * num_classes 
    dist_y_list = [0] * (num_classes - 1)
    for c1 in range(num_classes):
            for c2 in range(c1 + 1, num_classes):
                    dist_affinity_array[c2-c1] += mixing_matrix[c1,c2]
                    dist_x_list[c1] += mixing_matrix[c1,c2]
                    dist_y_list[c2-1] += mixing_matrix[c1,c2]
    dist_affinity_array = np.array(dist_affinity_array) / np.array(np.arange(num_classes,0, -1))
    dist_affinity_array = dist_affinity_array / sum(dist_affinity_array)
    dist_affinity_mean = 0
    for i in range(len(dist_affinity_array)):
            dist_affinity_mean += dist_affinity_array[i] * i
    dist_affinity_std = np.sqrt(np.cov(range(len(dist_affinity_array)), aweights=dist_affinity_array))

    dist_x_list = np.array(dist_x_list) / np.array(np.arange(num_classes,0, -1))
    dist_x_list = dist_x_list / sum(dist_x_list) 
    dist_x_mean = 0
    for i in range(len(dist_x_list)):
            dist_x_mean += dist_x_list[i] * i

    dist_y_list = np.array(dist_y_list) / np.array(np.arange(1, num_classes))
    dist_y_list = dist_y_list / sum(dist_y_list) 
    dist_y_mean = 0
    for i in range(len(dist_y_list)):
            dist_y_mean += dist_y_list[i] * i

    return dict(
        num_classes=num_classes, 
        homophily=homophily, 
        self_affinity_imbalance_ratio=self_affinity_imbalance_ratio, 
        interclass_affinity_imbalance_ratio=interclass_affinity_imbalance_ratio,
        dist_affinity_mean=dist_affinity_mean, 
        dist_affinity_std=dist_affinity_std,
        dist_x_mean=dist_x_mean, 
        dist_y_mean=dist_y_mean 
    )


@dataclass
class SkyMapMetrics:
    num_nodes: int
    density: float
    
    # Class distribution
    num_classes: int
    class_imbalance_ratio: float
    log_logistic_delta: float
    log_logistic_lambda: float
    
    # Mixing Matrix params.
    dist_x_mean: float
    dist_y_mean: float
    dist_affinity_mean: float
    self_affinity_imbalance_ratio: float
    interclass_affinity_imbalance_ratio: float
    homophily: float
    
    # Feature distribution metrics
    num_features: float
    zero_gen_means_mean: float
    zero_gen_means_var: float
    zero_gen_vars_mean: float
    zero_gen_vars_var: float
    
    # Other
    perc_edges_conc_mean: float
    perc_edges_conc_var: float
    perc_nodes_conc_mean: float
    perc_nodes_conc_var: float
    deg_mult_mean: float
    deg_mult_var: float

    @classmethod
    def from_graph(cls, g: nx.Graph):
        
        all_metrics = {
            **compute_basic_metrics(g),
            **compute_class_distribution_metrics(g),
            **compute_degree_distribution_metrics(g),
            **compute_feature_distribution_metrics(g),
            **compute_laplacian_matrix_metrics(g),
            **compute_mixing_matrix_metrics(g)
        }
        
        # Clean metrics to use only the ones defined in dataclass
        required = cls.__annotations__.keys()
        metrics = {k: v for k, v in all_metrics.items() if k in required}
        return cls(**metrics)
    
    @property
    def num_edges(self):
        # density = (2 * num_edges) / (num_nodes*(num_nodes-1))
        return self.density*self.num_nodes*(self.num_nodes-1)/2
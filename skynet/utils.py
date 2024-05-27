import networkx as nx
import numpy as np
from scipy.optimize import minimize
import math
import random
import pandas as pd
import sys
from scipy.stats import beta

def cmf_discrete_log_logistic(x, d, l):
    return 1 / (1 + ((x + 1) / l) ** -d)


def pmf_discrete_log_logistic(x, d, l):
    return cmf_discrete_log_logistic(x + 1, d, l) - cmf_discrete_log_logistic(x, d, l)


def log_pmf_discrete_log_logistic(x, d, l):
    return np.log(pmf_discrete_log_logistic(x, d, l))


def fit_degree_distribution(G):
    try:
        num_nodes = len(G.nodes)
        histogram = np.array(nx.degree_histogram(G)) / num_nodes
        degree_list = np.array([val for (_, val) in G.degree()])

        def neg_loglike(params):
            d, l = params
            res = -np.sum(log_pmf_discrete_log_logistic(degree_list, d, l))
            return res

        def least_square(params):
            d, l = params
            degrees = np.array(range(len(histogram)))
            res = np.sqrt(np.sum((pmf_discrete_log_logistic(degrees, d, l) - histogram) ** 2))
            return res

        mean_degree = len(G.edges) / len(G.nodes)

        res = minimize(
            neg_loglike, [3, mean_degree], method="Nelder-Mead", bounds=[(1, 1000), (0, num_nodes)]
        )
        return res["x"]
    except:
        raise ValueError("Could not fig log-logistic distribution to degrees")
        # return [0, 0]


def normalize_rho(marginals, corr_matrix):
    n_features = len(marginals)
    corrnorm_matrix = np.zeros([n_features, n_features])
    for i in range(n_features):
        for j in range(n_features):
            p = marginals[i]
            tita_1 = p * (1-p)
            q = marginals[j]
            tita_2 = q * (1-q)
            if i != j and p*q > 0:

                corr_min =  (max(0, p+q-1) - p*q)/\
                    math.sqrt(tita_1*tita_2)

                corr_max =  (min(p,q) - p*q)/\
                    math.sqrt(tita_1*tita_2)
                corrnorm_matrix[i][j] = (corr_matrix[i][j] - corr_min) / (corr_max - corr_min)
    return corrnorm_matrix

def calculate_joint_prob(node_data, num_features):
    matrix = np.zeros([num_features, num_features])
    for i in range(num_features):
        for j in range(num_features):
            matrix[i][j] = sum(np.array(node_data['x'].iloc[i]) * np.array(node_data['x'].iloc[j]))/ num_features
    return matrix

def moments_joint_prob(node_data):
    features = np.array(list(node_data['x'].iloc))
    probs = []
    corr_matrix = np.nan_to_num(np.corrcoef(features, rowvar=False))
    marginals = np.sum(features, axis=0) / len(features)
    zero_vars = [i for i, p in enumerate(marginals) if p == 0]
    normcorr_matrix = normalize_rho(marginals, corr_matrix)
    non_zero_corr_matrix = np.delete(np.delete(normcorr_matrix, zero_vars, axis=0), zero_vars, axis=1)

    probs = non_zero_corr_matrix[np.triu_indices_from(non_zero_corr_matrix, 1)]
    return corr_matrix, probs




def imbalance_distribution(k, ir):
    if k == 1: 
        return np.array([1])
    p = 1/k + (1 - 1/k) * ir
    dstr = np.append(imbalance_distribution(k-1,ir) * (1-p), p)
    return dstr

def imb_distr_fit(x):
    elements = len(x)
    if elements == 1:
        return (1,0,0)

    x_sorted = sorted(x, reverse=True)
    irs = np.empty(elements-1)
    for i in range(elements-1):
        k = elements-i
        rest = sum(x_sorted[i:])
        m = x_sorted[i] / rest if rest > 0 else 1
        irs[i] = (m - 1/k) / (1-1/k)
    importance = np.linspace(1,0, len(irs))
    irs_mean = sum(irs * importance) / sum(importance)
    return (
        elements, 
        irs_mean, 
        0 if elements == 2 else np.sqrt(np.sum((irs - [irs_mean]*(elements-1)) ** 2))
        #0 if elements == 2 else mean(irs[:-1] - irs[1:])
    )


def imb_distr_fit_likelihood(x):
    elements = len(x)
    if elements == 1:
        return 0
    
    x_sorted = sorted(x, reverse=True)
    generated = random.choices(range(elements), k=1000, weights=x_sorted)
    def neg_loglike(ir):
        distr = list(reversed(imbalance_distribution(elements, ir[0])))
        logs = np.log([distr[e] for e in generated])
        res = - np.sum(logs)
        return res


    res = minimize(neg_loglike, [0.1], method = 'Nelder-Mead', bounds=[(0,1)])
    return res['x'][0]



# Mixing matrix procs

def get_mixing_matrix(g: nx.Graph) -> np.ndarray:
    
    data_dict = [{"node": i[0], "y": i[1]["y"], "x": i[1]["x"]} for i in g.nodes.items()]
    node_data = pd.DataFrame.from_dict(data_dict)
    data_dict = [{"node": d[0], "k": d[1], "k_internal": node_data[node_data['node'].isin(g.neighbors(d[0]))]['y'].eq(node_data[node_data['node'].eq(d[0])]['y'].iloc[0]).sum() } for d in g.degree()]
    deg_data = pd.DataFrame.from_dict(data_dict)
    deg_node_data = pd.merge(node_data, deg_data, on="node")
    deg_node_data['x_per'] = deg_node_data.apply(lambda x: sum(x['x']) / len(x['x']), axis=1)

    classes = list(set(node_data["y"]))
    num_classes = len(classes)
    
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

    idx=np.flip(np.argsort(np.diag(mixing_matrix_abs)))
    mixing_matrix_abs = mixing_matrix_abs[idx,:][:,idx]

    mixing_matrix = mixing_matrix_abs / nx.number_of_edges(g)
    return mixing_matrix


def decompose_mixin_matrix(mixing_matrix):
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

    homophility = sum(self_affinity_list)
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

    return {
        "num_classes": num_classes, 
        "homophility": homophility, 
        "self_affinity_imbalance_ratio": self_affinity_imbalance_ratio, 
        "interclass_affinity_imbalance_ratio": interclass_affinity_imbalance_ratio,
        "dist_affinity_mean": dist_affinity_mean, 
        "dist_affinity_std":dist_affinity_std,
        "dist_x_mean": dist_x_mean, 
        "dist_y_mean": dist_y_mean 
    }

def distance(a,b):
    values = zip(a,b)
    percential_vector = np.array([abs(v[0] - v[1]) / max(abs(v[0]),abs(v[1])) for v in values])
    percential_vector = np.nan_to_num(percential_vector)
    return np.linalg.norm(percential_vector, ord=2)


# Feuature/property generator

def gen_beta_moments(mean, var, n):
    if mean == 1:
        return np.asarray([ 1 ] * n)
    elif mean == 0:
        return np.asarray([ 0 ] * n)
    elif var < sys.float_info.epsilon:
        return np.asarray([mean] * n)
    else:
        #return np.asarray(np.random.normal(mean, sqrt(var), n)).clip(0,1)
        #a=mean*(mean*(1-mean)/var-1)
        #b=a*(1-mean)/mean
        #a = ((1-mean)/var - 1/mean)* mean * mean
        #b = ((1-mean)/var - 1/mean) * mean * (1-mean)
        if var >= mean*(1-mean):
            c = 1e-8
        else:
            c=(mean*(1-mean))/var-1
        a=mean*c
        b=(1-mean)*c
        
        return np.asarray(beta.rvs(a, b, loc = 0, scale = 1, size=n))
    
# Other
    
def pmf_disc_beta(x, mean, var):
    available = sorted(list(set(x)))
    available.append(np.nan)
    x_1 = x+np.array(available)[[available.index(x_i)+1 for x_i in x]]
    c=(mean*(1-mean))/var-1
    a=mean*c
    b=(1-mean)*c
    return np.nan_to_num(beta.cdf(x_1 , a, b), nan=1) - beta.cdf(x, a, b)


from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
sys.path.append('pytorch_DGCNN')
from util import GNNGraph
import multiprocess as mp
from itertools import islice

import torch
import gudhi as gd
from gudhi.representations import PersistenceImage

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

class TDA_GNNGraph(GNNGraph):
    def __init__(self, g, label, node_tags=None, node_features=None, graph_features=None):
        super().__init__(g, label, node_tags, node_features)
        self.graph_features = graph_features

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, 
                    max_nodes_per_hop=None, node_information=None, graph_feature=None, multi_angle=False, no_parallel=False):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(
            val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, labels):
        g_list = []
        if no_parallel:
            for i, j, g_label in tqdm(zip(links[0], links[1],labels)):
                ind = (i,j)
                g, graph_label, n_labels, n_features, graph_features = subgraph_extraction_labeling(
                    ind, A, g_label, h,
                    max_nodes_per_hop=max_nodes_per_hop, 
                    node_information=node_information, 
                    graph_feature=graph_feature,
                    multi_angle = multi_angle
                )
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                g_list.append(TDA_GNNGraph(g, graph_label, n_labels, n_features, graph_features))
            return g_list
        else:
            # the parallel extraction code
            start = time.time()
            # pool = mp.Pool(mp.cpu_count())
            pool = mp.Pool(64)
            results = pool.map_async(
                parallel_worker, 
                [((i, j), A, g_label, h, max_nodes_per_hop, node_information, graph_feature, multi_angle) for i, j, g_label in zip(links[0], links[1],labels)]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pool.join()
            pbar.close()
            g_list = [TDA_GNNGraph(g, graph_label, n_labels, n_features, graph_features) for g, graph_label, n_labels, n_features, graph_features in results]
            max_n_label['value'] = max(
                max([max(n_labels) for _, _, n_labels, _, _ in results]), max_n_label['value']
            )
            end = time.time()
            del results
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs = None, None
    if train_pos and train_neg:
        links = np.concatenate((train_pos[0],train_neg[0]),axis=0), np.concatenate((train_pos[1],train_neg[1]),axis=0)
        labels = np.concatenate((np.ones_like(train_pos[0]),np.zeros_like(train_neg[0])),axis=0)
        train_graphs = helper(A, links, labels)
    if test_pos and test_neg:
        links = np.concatenate((test_pos[0],test_neg[0]),axis=0), np.concatenate((test_pos[1],test_neg[1]),axis=0)
        labels = np.concatenate((np.ones_like(test_pos[0]),np.zeros_like(test_neg[0])),axis=0)
        test_graphs = helper(A, links, labels)
    elif test_pos:
        test_graphs = helper(A, test_pos, np.ones_like(test_pos[0]))
    return train_graphs, test_graphs, max_n_label['value'], h

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def node_labeling_fitration(edge_index, z):
    edge_weight = torch.ones(edge_index.size(1))
    for i in range(len(edge_weight)):
        if max(z[edge_index[0][i]], z[edge_index[1][i]]) == 0:
            weight = 0
        else:
            weight = max(z[edge_index[0][i]], z[edge_index[1][i]]) + min(z[edge_index[0][i]], z[edge_index[1][i]])/max(z[edge_index[0][i]], z[edge_index[1][i]])
        edge_weight[i] = weight
    distance_matrix = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(z.size, z.size))

    matrix = distance_matrix.toarray()

    return matrix

def get_TDA_feature(matrix,option=['Image','Landscape']):
    death = matrix.max() + 5
    matrix[np.where(matrix == 0)] = death # replace death
    np.fill_diagonal(matrix,0)
    # get persistence diagram
    RipsM = gd.RipsComplex(distance_matrix = matrix, max_edge_length=death+1)
    RipsM_tree = RipsM.create_simplex_tree(max_dimension = 1)
    RipsM_tree.persistence()

    result={}

    if 'Image' in option:
        dg0 = RipsM_tree.persistence_intervals_in_dimension(0)
        dg0 = dg0[~np.isinf(dg0).any(axis=1)]
        l0 = PersistenceImage(resolution=[1,18],weight=lambda x:  1/np.log(x[1]+1)).fit_transform([np.array(dg0)])

        result['Image'] = torch.tensor(l0)

    del RipsM, RipsM_tree

    return result

def multi_hop_subgraph(src, dst, hop_pair, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                    directed=False, A_csc=None, seed: int = 1):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]

    #A가 오염되지 않게 copy로 계산!!!
    Adj=A.copy()
    #hop 계산시 target link 연결 끊고 시도 2023.0926
    Adj = Adj.tolil()
    Adj[src,dst]=0
    Adj[dst,src]=0
    Adj = Adj.tocsr()

    nodes_1=k_single_hop_subnodes(src, hop_pair[0], Adj, sample_ratio=sample_ratio, max_nodes_per_hop=max_nodes_per_hop, seed=seed)
    nodes_2=k_single_hop_subnodes(dst, hop_pair[1], Adj, sample_ratio=sample_ratio, max_nodes_per_hop=max_nodes_per_hop, seed=seed)

    # target node 젤 앞에 두기
    nodes=nodes+list(set(nodes_1).union(set(nodes_2))-set(nodes))

    sub_graph = Adj[nodes, :][:, nodes]

    return nodes, sub_graph

def get_multi_PI(subgraph, h):
    TDA_feature = get_PI(subgraph)
    for i in range(1,h+1):
        for j in range(0,i+1):
            if i==j:
                if not i==h:
                    _, subgraph_multi = multi_hop_subgraph(0,1,[i,j],subgraph)
                    multi_TDA_feature = get_PI(subgraph_multi)
                    TDA_feature = torch.concat([TDA_feature, multi_TDA_feature],dim=1)
                else:
                    pass
            else:
                _, subgraph_multi = multi_hop_subgraph(0,1,[i,j],subgraph)
                multi_TDA_feature = (1/2)*get_PI(subgraph_multi)
                _, subgraph_multi = multi_hop_subgraph(0,1,[j,i],subgraph)
                multi_TDA_feature += (1/2)*get_PI(subgraph_multi)
                multi_TDA_feature /= multi_TDA_feature.max()
                TDA_feature = torch.concat([TDA_feature, multi_TDA_feature],dim=1)
    return TDA_feature


def get_PI(subgraph):

    #오염되지 않도록 Copy하여 A 바꾸기!!
    adj=subgraph.copy()
    #######positive 먼저 계산 ######### subgraph의 adj는 target node가 0,1 번째 index에 위치함.
    adj = adj.tolil() 
    adj[0, 1] = 1
    adj[1, 0] = 1
    adj = adj.tocsr()
    u, v, r = ssp.find(adj)

    ####################degree 계산 (np.array)####################
    # node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    node_labeling = node_label(adj)
    node_labeling = give_deg_info(node_labeling, adj, 5)
    distance_matrix = node_labeling_fitration(edge_index, node_labeling)
    result= get_TDA_feature(distance_matrix,option=['Image'])
    data1_TDA_feature = result['Image']

    ####### negative 계산 #########
    adj = adj.tolil() 
    adj[0, 1] = 0
    adj[1, 0] = 0
    adj = adj.tocsr()
    u, v, r = ssp.find(adj)

    # node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    node_labeling = node_label(adj)
    node_labeling = give_deg_info(node_labeling, adj, 5)

    distance_matrix = node_labeling_fitration(edge_index, node_labeling)
    result= get_TDA_feature(distance_matrix,option=['Image'])
    data2_TDA_feature = result['Image']

    TDA_feature = torch.concat([data1_TDA_feature, data2_TDA_feature],dim=1)
    TDA_feature /= TDA_feature.max()

    return TDA_feature

def k_single_hop_subnodes(src, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, seed: int = 1):
    nodes = [src]
    visited = set([src]) 
    fringe = set([src]) 

    for dist in range(1, num_hops+1):

        fringe = neighbors(fringe, A)

        fringe = fringe - visited
        fringe = fringe - set([src]) # 추가 파트

        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            set_random_seed(seed)
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)

    return nodes

def subgraph_extraction_labeling(ind, A, g_label, h=1, max_nodes_per_hop=None,
                                 node_information=None, graph_feature=None, multi_angle=False):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    graph_features = None
    if graph_feature is not None:
        if multi_angle:
            graph_features = get_multi_PI(subgraph, h)
        else:
            graph_features = get_PI(subgraph)
    # construct nx graph
    # g = nx.from_scipy_sparse_matrix(subgraph)
    g = nx.from_scipy_sparse_array(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, g_label, labels.tolist(), features, graph_features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


# degree node labeling 추가하기
def give_deg_info(z, adj, Max_deg):
    graph = adj.toarray()
    deg = np.sum(graph, axis=1)
    deg = np.clip(deg, 0, Max_deg)
    z = z*Max_deg + Max_deg - deg

    return z

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels


def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


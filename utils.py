# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import networkx as nx
import pickle
import parmap
import time

import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, to_undirected
from torch_geometric.utils import (from_scipy_sparse_matrix, is_undirected, negative_sampling, 
                                   add_self_loops, train_test_split_edges)
import pdb
import gudhi as gd
from gudhi.representations import PersistenceImage

cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_edges(data, seed=123, val_ratio=0.05,test_ratio=0.1, practical_neg_sample=False):
    set_random_seed(seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v= floor(val_ratio * row.size(0)).int() #number of validation positive edges
    n_t=floor(test_ratio * row.size(0)).int() #number of test positive edges
    #split positive edges   
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    #sample negative edges
    if practical_neg_sample == False:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:row.size(0)]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
        data.train_neg = torch.stack([row, col], dim=0)

    else:

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        data.test_neg = torch.stack([neg_row, neg_col], dim=0)

        # Sample the train and val negative edges with only knowing 
        # the train positive edges
        row, col = data.train_pos
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

        n_tot = n_v + data.train_pos.size(1)
        perm = torch.randperm(neg_row.size(0))[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:], neg_col[n_v:]
        data.train_neg = torch.stack([row, col], dim=0)

    return data


def load_unsplitted_data(data_name):
    # read .mat format files
    data_dir = os.path.join(par_dir, 'data/no_feature/{}.mat'.format(data_name))
    print('Load data from: '+ data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = torch.max(data.edge_index)+1
    return data

def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def single_number_hop_subnodes(src, num_limit, A, seed: int = 1):
    nodes = [src]
    visited = set([src]) 
    fringe = set([src]) 

    if num_limit == 0: #num_limit = 0 인 경우 처리 
        return []

    while 1:
        fringe = neighbors(fringe, A)

        fringe = fringe - visited
        fringe = fringe - set([src])

        visited = visited.union(fringe)

        if len(nodes)+len(fringe) >= num_limit:
            set_random_seed(seed)
            fringe = random.sample(fringe, num_limit-len(nodes))
            nodes = nodes + list(fringe)
            break

        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)

    return nodes

def k_single_hop_subnodes(src, num_hops, starting_hop_restric, A, seed: int = 1):
    
    nodes = [src]
    visited = set([src]) 
    fringe = set([src]) 


    for iterations in range(1, num_hops+1):

        if iterations >= starting_hop_restric[0]:
            max_nodes_per_hop = starting_hop_restric[1]
        else:
            max_nodes_per_hop= None

        fringe = neighbors(fringe, A)

        fringe = fringe - visited
        fringe = fringe - set([src]) 

        visited = visited.union(fringe)

        if max_nodes_per_hop is not None:
            set_random_seed(seed)
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)

    return nodes


def union_single_hop(src, num_hops, num_limit, starting_hop_restric, A, seed: int = 1):
    nodes1 = single_number_hop_subnodes(src, num_limit, A, seed= seed)
    nodes2 = k_single_hop_subnodes(src, num_hops, starting_hop_restric, A, seed= seed)

    return [src]+list(set(nodes1[1:]+nodes2[1:]))

def multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=None, seed: int = 1):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]

    #A가 오염되지 않게 copy로 계산!!!
    Adj=A.copy()
  
    #hop 계산시 target link 연결 끊고 시도 2023.0926
    Adj = Adj.tolil()  
    Adj[src,dst]=0
    Adj[dst,src]=0
    Adj = Adj.tocsr()

    nodes_1=union_single_hop(src, hop_pair[0], num_limit_pair[0], starting_hop_restric, Adj, seed=seed)
    nodes_2=union_single_hop(dst, hop_pair[1], num_limit_pair[1], starting_hop_restric, Adj, seed=seed)

    # target node 젤 앞에 두기
    nodes=nodes+list(set(nodes_1).union(set(nodes_2))-set(nodes))
    subgraph = Adj[nodes, :][:, nodes]

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, node_features, y

def multihop_extract_enclosing_subgraphs(lst, A, x, y, node_label='drnl', starting_hop_restric=[3,100],
                                deg_cut=0.95, Max_hops=2, multi_angle=False, angle_hop='[3,0]', step_size=100, seed: int = 1, degree_info = False):
    src, dst = lst
    graph = A.toarray()
    deg = np.sum(graph, axis=1)
    
    if deg_cut < 1:
        Max_deg = int(np.quantile(deg, deg_cut))
    elif deg_cut >= 1:
        if type(deg_cut) != int:
            raise Exception("If deg_cut >= 1, it must be integer")
        Max_deg=deg_cut

    if node_label == 'degdrnl':
        node_label = 'drnl'
        degree_info = True
    else:
        degree_info = False

    if multi_angle:
        multi_data=[]

        for i in range(1,Max_hops+1):
            for j in range(i+1):
                num_limit_pair = [i*step_size, j*step_size]
                hop_pair=[i,j]

                if i == j:
                    tmp = multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=x, seed=seed)
                    data = [construct_pyg_graph(*tmp, Max_deg=Max_deg, node_label=node_label, degree_info=degree_info)]
                    data[0][0].hop = hop_pair
                else: 
                    tmp = multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=x, seed=seed)
                    data1 = construct_pyg_graph(*tmp, Max_deg=Max_deg, node_label=node_label, degree_info=degree_info)

                    num_limit_pair = [j*step_size, i*step_size]
                    hop_pair = [j,i]

                    tmp = multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=x, seed=seed)
                    data2 = construct_pyg_graph(*tmp, Max_deg=Max_deg, node_label=node_label, degree_info=degree_info)
                    data1[0].hop = hop_pair
                    data = [data1, data2]
                multi_data.append(data)
        multi_data[0][0][0].target_nodes = torch.tensor([src,dst])
    else:
        multi_data=[]
        i,j = int(angle_hop[1]), int(angle_hop[3])
        num_limit_pair = [i*step_size, j*step_size]
        hop_pair=[i,j]

        if i == j:
            tmp = multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=x, seed=seed)
            data = [construct_pyg_graph(*tmp, Max_deg=Max_deg, node_label=node_label, degree_info=degree_info)]
            data[0][0].hop = hop_pair
        else: 
            tmp = multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=x, seed=seed)
            data1 = construct_pyg_graph(*tmp, Max_deg=Max_deg, node_label=node_label, degree_info=degree_info)

            num_limit_pair = [j*step_size, i*step_size]
            hop_pair = [j,i]

            tmp = multi_hop_subgraph(src, dst, hop_pair, num_limit_pair, starting_hop_restric, A, y, node_features=x, seed=seed)
            data2 = construct_pyg_graph(*tmp, Max_deg=Max_deg, node_label=node_label, degree_info=degree_info)
            data1[0].hop = hop_pair
            data = [data1, data2]
        multi_data.append(data)

    return multi_data


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).

    graph = adj.toarray()

    dist2src = shortest_path(graph, directed=False, unweighted=True, indices=src)
    # dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(graph, directed=False, unweighted=True, indices=dst)
    # dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.

    z[torch.isnan(z)]=0.

    return z.to(torch.long)


def give_deg_info(z, adj, Max_deg):
    graph = adj.toarray()
    deg = np.sum(graph, axis=1)
    deg = np.clip(deg, 0, Max_deg)
    z = z*Max_deg + Max_deg - deg

    return z

######### 오염 안되게 copy 파트 왜 없앰 ? 
def construct_pyg_graph(node_ids, A, node_features, y, Max_deg, node_label='drnl', degree_info=False):

    #오염되지 않도록 Copy하여 A 바꾸기!!
    adj=A.copy()
    #######positive 먼저 계산 ######### subgraph의 adj는 target node가 0,1 번째 index에 위치함.
    adj = adj.tolil()  
    adj[0, 1] = 1
    adj[1, 0] = 1
    adj = adj.tocsr()
    u, v, r = ssp.find(adj)

    ####################degree 계산 (np.array)####################
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)

    y = torch.tensor([y])

    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    else:
        raise Exception("Something Wrong in construct_pyg_graph")

    if degree_info == True:
        z = give_deg_info(z, adj, Max_deg)

    data_pos = Data(edge_index=edge_index, y=y, z=z)

    ####### negative 계산 #########
    adj = adj.tolil()  
    adj[0, 1] = 0
    adj[1, 0] = 0
    adj = adj.tocsr()
    u, v, r = ssp.find(adj)

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)

    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    else:
        raise Exception("Something Wrong in construct_pyg_graph")
 
    if degree_info == True:
        z = give_deg_info(z, adj, Max_deg)
    
    # data_neg = Data(edge_index=edge_index, y=y, z=z, node_id=node_ids)
    data_neg = Data(edge_index=edge_index, y=y, z=z)

    return data_neg, data_pos



def node_labeling_fitration(edge_index, z):
    edge_weight = torch.ones(edge_index.size(1))
    for i in range(len(edge_weight)):
        weight = max(z[edge_index[0][i]], z[edge_index[1][i]]) + min(z[edge_index[0][i]], z[edge_index[1][i]])/max(z[edge_index[0][i]], z[edge_index[1][i]])
        edge_weight[i] = weight
    distance_matrix = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(z.size(0), z.size(0)))

    matrix = distance_matrix.toarray()
    
    return matrix

def get_TDA_feature_image(matrix, onedim_PH=False):
    death = matrix.max()*1.2
    matrix[np.where(matrix == 0)] = death # replace death
    np.fill_diagonal(matrix,0)
    
    # get persistence diagram
    RipsM = gd.RipsComplex(distance_matrix = matrix, max_edge_length=death+1)
    if onedim_PH:
        RipsM_tree = RipsM.create_simplex_tree(max_dimension = 2)
        RipsM_tree.persistence()

        # get persistence Image
        dg0 = RipsM_tree.persistence_intervals_in_dimension(0)
        dg0 = dg0[~np.isinf(dg0).any(axis=1)]
    
        dg1 = RipsM_tree.persistence_intervals_in_dimension(1)
        dg0[np.where(dg0 == np.inf)] = death
        if len(dg1) == 0:
            dg1 = np.empty([0,2])

        l0 = PersistenceImage(resolution=[16,16],weight=lambda x:  1/np.log(x[1]+1)).fit_transform([np.array(dg0)]) ## 웨이트 이렇게 확정 ? ?
        l1 = PersistenceImage(resolution=[16,16],weight=lambda x:  1/np.log(x[0]+1)).fit_transform([np.array(dg1)]) 
    
        return torch.tensor(np.concatenate((l0,l1),axis=1))
    
    else:
        RipsM_tree = RipsM.create_simplex_tree(max_dimension = 1)
        RipsM_tree.persistence()

        # get persistence Image
        dg0 = RipsM_tree.persistence_intervals_in_dimension(0)
        dg0 = dg0[~np.isinf(dg0).any(axis=1)]
    
        dg0[np.where(dg0 == np.inf)] = death

        l0 = PersistenceImage(resolution=[16,16],weight=lambda x:  1/np.log(x[1]+1)).fit_transform([np.array(dg0)]) ## 웨이트 이렇게 확정 ? ?
    
        return torch.tensor(l0)

def multi_get_PI(full_data, onedim_PH):
    base_data = full_data[0][0][0] # first data1
    
    if onedim_PH:
        delta_TDA_feature = torch.empty((0,1024))
    else:
        delta_TDA_feature = torch.empty((0,512))
    for multi_data in full_data:
        value = 0
        for data in multi_data:
            data1, data2 = data
            distance_matrix = node_labeling_fitration(data1.edge_index, data1.z)
            data1_TDA_feature =  get_TDA_feature_image(distance_matrix, onedim_PH)

            distance_matrix = node_labeling_fitration(data2.edge_index, data2.z)
            data2_TDA_feature =  get_TDA_feature_image(distance_matrix, onedim_PH)

            value += torch.concat([data1_TDA_feature, data2_TDA_feature],dim=1)
        delta_TDA_feature = torch.concat([delta_TDA_feature, value],dim=0)

    base_data.delta_TDA_feature = delta_TDA_feature
    
    return base_data

def Calculate_TDA_feature(data_name, starting_hop_restric, node_label, deg_cut, seed, Max_hops, multi_angle, angle_hop, onedim_PH, num_cpu, multiprocess, **kwargs):
    if data_name in ['USAir', 'NS', 'Celegans','Power','Router','Yeast','PB','Ecoli']:
        data = load_unsplitted_data(data_name)

    data = split_edges(data)
    edge_weight = torch.ones(data.train_pos.size(1)*2, dtype=int)
    train_edge = torch.concat((data.train_pos,data.train_pos[[1,0]]),dim=1)
    A = ssp.csr_matrix((edge_weight, (train_edge[0],train_edge[1])), shape=(data.num_nodes, data.num_nodes))

    if onedim_PH:
        if multi_angle:
            dir_ = cur_dir + '/data_TDA/multi_angle/' + data_name + '/seed' + str(seed)
        else:
            dir_ = cur_dir + '/data_TDA/single_angle/' + data_name + '/seed' + str(seed)
    else:
        if multi_angle:
            dir_ = cur_dir + '/data_TDA/multi_angle(0dim)/' + data_name + '/seed' + str(seed)
        else:
            dir_ = cur_dir + '/data_TDA/single_angle(0dim)/' + data_name + '/seed' + str(seed)
    # 없으면 폴더 만들기
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    # # num_cpu = int(os.cpu_count()/2) ############################### 이 부분도 파라미터로 받을 수 있으면 좋을 것 같다. 
    # num_cpu = 1 #임시로 지정
    # multiprocess=True

    if os.path.exists(dir_+'/val_dataset_TDA.pickle'):
        pass
    else:
        pos_edge, neg_edge = data.val_pos, data.val_neg

        #데이터별 step size 계산
        degree = A.toarray().sum(axis=0)
        alpha = (degree[data.train_pos.transpose(0,1)].sum() + degree[data.val_pos.transpose(0,1)].sum()) / (degree[data.train_neg.transpose(0,1)].sum() + degree[data.val_neg.transpose(0,1)].sum())
        num_supplement = data.num_nodes/(alpha-1)**2/400 #0.5 %
        step_size=round(int(num_supplement))

        if multiprocess == True:
            print('Start extract positive subgraphs: valid')
            valid_pos_list = parmap.starmap(multihop_extract_enclosing_subgraphs, [[lst] for lst in pos_edge.t().tolist()],  
                A=A, x=data.x, y=1, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size,seed=seed,pm_pbar=True, pm_processes=num_cpu)
            print('Start extract negative subgraphs: valid')
            valid_neg_list = parmap.starmap(multihop_extract_enclosing_subgraphs, [[lst] for lst in neg_edge.t().tolist()],  
                A=A, x=data.x, y=0,  node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size,seed=seed,pm_pbar=True, pm_processes=num_cpu)
            val_dataset = valid_pos_list + valid_neg_list
            print('Calculate persistence image: valid')
            val_dataset_TDA = parmap.starmap(multi_get_PI, [[graphdata] for graphdata in val_dataset], onedim_PH=onedim_PH, pm_pbar=True, pm_processes=num_cpu)
        else:
            valid_pos_list = [multihop_extract_enclosing_subgraphs(lst, A=A, x=data.x, y=1, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size, seed=seed) for lst in tqdm(pos_edge.t().tolist(), desc = 'Extract positive subgraphs: valid')]
            valid_neg_list = [multihop_extract_enclosing_subgraphs(lst, A=A, x=data.x, y=0, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size, seed=seed) for lst in tqdm(neg_edge.t().tolist(), desc = 'Extract negative subgraphs: valid')]
            val_dataset = valid_pos_list + valid_neg_list
            val_dataset_TDA = [multi_get_PI(graphdata, onedim_PH=onedim_PH) for graphdata in tqdm(val_dataset, desc= 'Calculate persistence image: valid')]

        with open(dir_+'/val_dataset_TDA.pickle', 'wb') as f:
            pickle.dump(val_dataset_TDA, f, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(dir_+'/test_dataset_TDA.pickle'):
        pass
    else:
        pos_edge, neg_edge = data.test_pos, data.test_neg

        #데이터별 step size 계산
        degree = A.toarray().sum(axis=0)
        alpha = (degree[data.train_pos.transpose(0,1)].sum() + degree[data.val_pos.transpose(0,1)].sum()) / (degree[data.train_neg.transpose(0,1)].sum() + degree[data.val_neg.transpose(0,1)].sum())
        num_supplement = data.num_nodes/(alpha-1)**2/400 #0.5 %
        step_size=round(int(num_supplement))    

        if multiprocess == True:
            print('Start extract positive subgraphs: test')
            test_pos_list = parmap.starmap(multihop_extract_enclosing_subgraphs, [[lst] for lst in pos_edge.t().tolist()],  
                A=A, x=data.x, y=1, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size,seed=seed,pm_pbar=True, pm_processes=num_cpu)
            print('Start extract positive subgraphs: test')
            test_neg_list = parmap.starmap(multihop_extract_enclosing_subgraphs, [[lst] for lst in neg_edge.t().tolist()],  
                A=A, x=data.x, y=0, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size,seed=seed,pm_pbar=True, pm_processes=num_cpu)
            test_dataset = test_pos_list + test_neg_list
            print('Calculate persistence image: test')
            test_dataset_TDA = parmap.starmap(multi_get_PI, [[graphdata] for graphdata in test_dataset], onedim_PH=onedim_PH, pm_pbar=True, pm_processes=num_cpu)
        else:
            test_pos_list = [multihop_extract_enclosing_subgraphs(lst, A=A, x=data.x, y=1, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size, seed=seed) for lst in tqdm(pos_edge.t().tolist(), desc = 'Extract positive subgraphs: test')]
            test_neg_list = [multihop_extract_enclosing_subgraphs(lst, A=A, x=data.x, y=0, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size, seed=seed) for lst in tqdm(neg_edge.t().tolist(), desc = 'Extract negative subgraphs: test')]
            test_dataset = test_pos_list + test_neg_list
            test_dataset_TDA = [multi_get_PI(graphdata, onedim_PH=onedim_PH) for graphdata in tqdm(test_dataset, desc= 'Calculate persistence image: test' )]   

        with open(dir_+'/test_dataset_TDA.pickle', 'wb') as f:
            pickle.dump(test_dataset_TDA, f, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(dir_+'/train_dataset_TDA.pickle'):
        pass
    else:
        pos_edge, neg_edge = data.train_pos, data.train_neg
        #데이터별 step size 계산
        degree = A.toarray().sum(axis=0)
        alpha = (degree[data.train_pos.transpose(0,1)].sum() + degree[data.val_pos.transpose(0,1)].sum()) / (degree[data.train_neg.transpose(0,1)].sum() + degree[data.val_neg.transpose(0,1)].sum())
        num_supplement = data.num_nodes/(alpha-1)**2/400 #0.5 %
        step_size=round(int(num_supplement))

        if multiprocess == True:
            print('Start extract positive subgraphs: train')
            train_pos_list = parmap.starmap(multihop_extract_enclosing_subgraphs, [[lst] for lst in pos_edge.t().tolist()],  
                A=A, x=data.x, y=1, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size,seed=seed,pm_pbar=True, pm_processes=num_cpu)
            print('Start extract positive subgraphs: train')
            train_neg_list = parmap.starmap(multihop_extract_enclosing_subgraphs, [[lst] for lst in neg_edge.t().tolist()],  
                A=A, x=data.x, y=0, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size,seed=seed,pm_pbar=True, pm_processes=num_cpu)
            train_dataset = train_pos_list + train_neg_list
            print('Calculate persistence image: train')
            train_dataset_TDA = parmap.starmap(multi_get_PI, [[graphdata] for graphdata in train_dataset], onedim_PH=onedim_PH, pm_pbar=True, pm_processes=num_cpu)
        else:
            train_pos_list = [multihop_extract_enclosing_subgraphs(lst, A=A, x=data.x, y=1, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size, seed=seed) for lst in tqdm(pos_edge.t().tolist(), desc = 'Extract positive subgraphs: train')]
            train_neg_list = [multihop_extract_enclosing_subgraphs(lst, A=A, x=data.x, y=0, node_label=node_label, starting_hop_restric=starting_hop_restric, deg_cut=deg_cut, Max_hops=Max_hops, multi_angle=multi_angle, angle_hop=angle_hop, step_size=step_size, seed=seed) for lst in tqdm(neg_edge.t().tolist(), desc = 'Extract negative subgraphs: train')]
            train_dataset = train_pos_list + train_neg_list
            train_dataset_TDA = [multi_get_PI(graphdata, onedim_PH=onedim_PH) for graphdata in tqdm(train_dataset, desc= 'Calculate persistence image: train')]   

        with open(dir_+'/train_dataset_TDA.pickle', 'wb') as f:
            pickle.dump(train_dataset_TDA, f, pickle.HIGHEST_PROTOCOL)

    print(f'{data_name} is DONE !!')

def load_TDA_data(data_name, onedim_PH, multi_angle, seed, batch_size=1024):

    if onedim_PH:
        if multi_angle:
            dir_ = cur_dir + '/data_TDA/multi_angle/' + data_name + '/seed' + str(seed)
        else:
            dir_ = cur_dir + '/data_TDA/single_angle/' + data_name + '/seed' + str(seed)
    else:
        if multi_angle:
            dir_ = cur_dir + '/data_TDA/multi_angle(0dim)/' + data_name + '/seed' + str(seed)
        else:
            dir_ = cur_dir + '/data_TDA/single_angle(0dim)/' + data_name + '/seed' + str(seed)

    # load
    with open(osp.join(dir_,'train_dataset_TDA.pickle'), 'rb') as f:
        train_dataset_TDA = pickle.load(f)
    maxs = torch.empty((0,train_dataset_TDA[0].delta_TDA_feature.size(0)))
    image_size = train_dataset_TDA[0].delta_TDA_feature.size(1)
    if multi_angle:
        image_shape = (1,maxs.size(1),image_size)
    else:
        image_shape = (1,image_size)
    for data in train_dataset_TDA:
        feature_abs = torch.abs(data.delta_TDA_feature) 
        max_ = feature_abs.max(dim=1)[0].reshape((1,maxs.size(1)))
        maxs = torch.cat((maxs, max_), 0)
    M = maxs.max(dim=0)[0]
    for data in train_dataset_TDA:
        nomalized_feature = ((1/M).unsqueeze(1) * (data.delta_TDA_feature))
        data.delta_TDA_feature = nomalized_feature.reshape(image_shape)
    
    with open(osp.join(dir_,'val_dataset_TDA.pickle'), 'rb') as f:
        val_dataset_TDA = pickle.load(f)
    for data in val_dataset_TDA:
        nomalized_feature = ((1/M).unsqueeze(1) * (data.delta_TDA_feature))
        data.delta_TDA_feature = nomalized_feature.reshape(image_shape)
    
    with open(osp.join(dir_,'test_dataset_TDA.pickle'), 'rb') as f:
        test_dataset_TDA = pickle.load(f)
    for data in test_dataset_TDA:
        nomalized_feature = ((1/M).unsqueeze(1) * (data.delta_TDA_feature))
        data.delta_TDA_feature = nomalized_feature.reshape(image_shape)

    train_loader = DataLoader(train_dataset_TDA, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_TDA, batch_size=batch_size)
    test_loader = DataLoader(test_dataset_TDA, batch_size=batch_size)

    return train_loader, val_loader, test_loader

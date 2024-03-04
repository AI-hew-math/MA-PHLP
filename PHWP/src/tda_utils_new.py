###TDA Library
import gudhi as gd
from gudhi.representations import Landscape
from gudhi.representations import PersistenceImage
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

import torch
import numpy as np

###############

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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


def node_labeling_fitration(edge_index, z):
    edge_weight = torch.ones(edge_index.size(1))
    for i in range(len(edge_weight)):
        if max(z[edge_index[0][i]], z[edge_index[1][i]]) == 0:
            weight = 0
        else:
            weight = max(z[edge_index[0][i]], z[edge_index[1][i]]) + min(z[edge_index[0][i]], z[edge_index[1][i]])/max(z[edge_index[0][i]], z[edge_index[1][i]])

        edge_weight[i] = weight
    distance_matrix = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(z.size(0), z.size(0)))

    matrix = distance_matrix.toarray()

    return matrix


def get_TDA_feature(matrix, args): # Not multi = 36 / Multi = 18
    death = matrix.max() + 5
    matrix[np.where(matrix == 0)] = death # replace death
    np.fill_diagonal(matrix,0)
    # get persistence diagram
    RipsM = gd.RipsComplex(distance_matrix = matrix, max_edge_length=death+1)
    RipsM_tree = RipsM.create_simplex_tree(max_dimension = 1)
    RipsM_tree.persistence()

    # get persistence image
    dg0 = RipsM_tree.persistence_intervals_in_dimension(0)
    dg0[np.where(dg0 == np.inf)] = death

    if args.multi_angle == True:
        feature_size=18
    else:
        feature_size=36
    l0 = PersistenceImage(resolution=[1,feature_size],weight=lambda x:  1/np.log(x[1]+1)).fit_transform([np.array(dg0)])

    return torch.tensor(l0)

def give_deg_info(z, adj, Max_deg):
    graph = adj.toarray()
    deg = np.sum(graph, axis=1)
    deg = np.clip(deg, 0, Max_deg)
    
    z = z*Max_deg + Max_deg - deg

    return z


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res

def k_single_hop_subnodes(src, num_hops, A, max_nodes_per_hop=None, seed: int = 1):
    nodes = [src]
    visited = set([src]) 
    fringe = set([src]) 

    for dist in range(1, num_hops+1):

        fringe = neighbors(fringe, A)

        fringe = fringe - visited
        fringe = fringe - set([src])

        visited = visited.union(fringe)

        if max_nodes_per_hop is not None:
            set_random_seed(seed)
            if max_nodes_per_hop < len(fringe):
                fringe = np.random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)

    return nodes


def multi_hop_submask(hop_pair, subgraph, max_nodes_per_hop=None, seed: int = 1):

    num_nodes =subgraph.x.size(0)
    adjacency_data = np.ones(subgraph.edge_index.size(1)) 
    adjacency_matrix = csr_matrix((adjacency_data, (subgraph.edge_index[0], subgraph.edge_index[1])), shape=(num_nodes, num_nodes))

    src, dst = map(int, subgraph.edge_index[:,~subgraph.edge_mask][0])
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]

    Adj=adjacency_matrix.copy()
    
    Adj=Adj.tolil()
    Adj[src,dst]=0
    Adj[dst,src]=0
    Adj=Adj.tocsr()

    nodes_1=k_single_hop_subnodes(src, hop_pair[0], Adj, max_nodes_per_hop=max_nodes_per_hop, seed=seed)
    nodes_2=k_single_hop_subnodes(dst, hop_pair[1], Adj, max_nodes_per_hop=max_nodes_per_hop, seed=seed)

    selected_nodes = list(set(nodes_1).union(set(nodes_2)))

    mask = np.ones(Adj.shape[0], dtype=bool)  
    mask[selected_nodes] = False  

    return mask


def get_PI(subgraph,args,multi_submask = None):

    num_nodes =subgraph.x.size(0)
    adjacency_data = np.ones(subgraph.edge_index.size(1)) 
    adjacency_matrix = csr_matrix((adjacency_data, (subgraph.edge_index[0], subgraph.edge_index[1])), shape=(num_nodes, num_nodes))

    if callable(multi_submask):
        adj_lil = adjacency_matrix.tolil()
        adj_lil[multi_submask, :] = 0  
        adj_lil[:, multi_submask] = 0  
        adjacency_matrix = adj_lil.tocsr()

    src,dst = subgraph.edge_index[:,~subgraph.edge_mask][0] 
 
    ###### positive #########
    adj=adjacency_matrix.copy()
    adj=adj.tolil()
    adj[src,dst]=1 
    adj[dst,src]=1
    adj=adj.tocsr()

    u, v, r = ssp.find(adj)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)

    z = drnl_node_labeling(adj, src, dst)
    # z = drnl_node_labeling(adj, 0, 1)
    # if num_nodes < 4:
    #     z = drnl_node_labeling(adj, 0, 1)
    # else:
    #     z = drnl_node_labeling(adj, 2, 3)
    inf_value = torch.max(z) + 1

    node_labeling = give_deg_info(z, adj, 5)

    distance_matrix = node_labeling_fitration(edge_index, node_labeling)
    data1_TDA_feature = get_TDA_feature(distance_matrix, args)


    ####### negative #########
    adj=adjacency_matrix.copy()

    adj=adj.tolil()
    adj[src,dst]=0 
    adj[dst,src]=0
    adj=adj.tocsr()

    u, v, r = ssp.find(adj)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)

    z = drnl_node_labeling(adj, src, dst)
    # z = drnl_node_labeling(adj, 0, 1)
    # if num_nodes < 4:
    #     z = drnl_node_labeling(adj, 0, 1)
    # else:
    #     z = drnl_node_labeling(adj, 2, 3)
    z[torch.where(z==0)]=inf_value

    node_labeling = give_deg_info(z, adj, 5)

    distance_matrix = node_labeling_fitration(edge_index, node_labeling)
    data2_TDA_feature= get_TDA_feature(distance_matrix, args)
 
    TDA_feature = torch.concat([data1_TDA_feature, data2_TDA_feature],dim=1)
    TDA_feature /= TDA_feature.max()

    return TDA_feature.to(torch.float)


def get_multi_PI(subgraph, args): 

    if args.multi_angle==True:
        TDA_feature = get_PI(subgraph, args)

        # for i in range(args.max_hop):
        #     subgraph_multi_mask = multi_hop_submask([i,args.max_hop],subgraph)
        #     multi_TDA_feature = (1/2)*get_PI(subgraph, args, subgraph_multi_mask)
        #     subgraph_multi_mask = multi_hop_submask([args.max_hop,i],subgraph)
        #     multi_TDA_feature += (1/2)*get_PI(subgraph, args, subgraph_multi_mask)

        #     # Normalization
        #     half_size= multi_TDA_feature.size(1)//2
        #     first_half = multi_TDA_feature[0, :half_size] / multi_TDA_feature[0, :half_size].max()
        #     second_half = multi_TDA_feature[0, half_size:] / multi_TDA_feature[0, half_size:].max()
        #     multi_TDA_feature = torch.concat([first_half, second_half],dim=0).unsqueeze(0)
            
        #     TDA_feature = torch.concat([TDA_feature, multi_TDA_feature],dim=1)

        # for i in range(1, args.max_hop):
        #     subgraph_multi_mask = multi_hop_submask([i,i],subgraph)
        #     multi_TDA_feature = get_PI(subgraph, args, subgraph_multi_mask)
            
        #     TDA_feature = torch.concat([TDA_feature, multi_TDA_feature],dim=1)

        for i in range(1,args.max_hop+1):
            for j in range(0,i+1):
                if i==j:
                    if not i==args.max_hop:
                        subgraph_multi_mask = multi_hop_submask([i,j],subgraph)
                        multi_TDA_feature = get_PI(subgraph, args, subgraph_multi_mask)
                        
                        TDA_feature = torch.concat([TDA_feature, multi_TDA_feature],dim=1)
                else:
                    subgraph_multi_mask = multi_hop_submask([i,j],subgraph)
                    multi_TDA_feature = (1/2)*get_PI(subgraph, args, subgraph_multi_mask)
                    subgraph_multi_mask = multi_hop_submask([j,i],subgraph)
                    multi_TDA_feature += (1/2)*get_PI(subgraph, args, subgraph_multi_mask)
                    multi_TDA_feature /= multi_TDA_feature.max()

                    TDA_feature = torch.concat([TDA_feature, multi_TDA_feature],dim=1)


    else:
        TDA_feature = get_PI(subgraph, args)

    subgraph.tda_vector = TDA_feature
    return subgraph

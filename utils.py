import torch
import os
from collections import OrderedDict
import torch
import math
from statistics import median, mean
import random
import numpy as np
import networkx as nx
import copy
from torch_geometric.utils import to_networkx
from networkx.generators.random_graphs import erdos_renyi_graph
from math import inf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import scipy.sparse as sp
import sklearn.preprocessing as preprocessing
from scipy.sparse import linalg
import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.multiclass import OneVsRestClassifier
from copy import deepcopy
from torch_scatter import scatter


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not int(0), parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)
    return total_norm

def filter_state_dict(state_dict,name):
    keys_to_del = []
    for key in state_dict.keys():
        if name not in key:
            keys_to_del.append(key)
    for key in sorted(keys_to_del, reverse=True):
        del state_dict[key]
    return state_dict


def monitor_grad_norm(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def monitor_grad_norm_2(gradients):
    total_norm = 0
    for p in gradients:
        if p is not int(0):
            param_norm = p.data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def monitor_weight_norm(model):
    parameters = list(filter(lambda p: p is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index,weights):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index,weights)
    return model.test(z, pos_edge_index, neg_edge_index)


def test2(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def test_one(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index,weights):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def global_test(args, model, data_batch, weights):
    model.eval()
    auc_list, ap_list = [], []
    for data in data_batch:
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data.batch = None
        # Test Ratio is Fixed at 0.1
        meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
        if args.use_fixed_feats and args.dataset=='REDDIT-MULTI-12K':

            num_nodes = data.num_nodes
            perm = torch.randperm(args.feats.size(0))
            perm_idx = perm[:num_nodes]
            data.x = args.feats[perm_idx]
        elif args.use_same_fixed_feats and args.dataset=='REDDIT-MULTI-12K':
            node_feats = args.feats[0].unsqueeze(0).repeat(num_nodes,1)
            data.x = node_feats
        try:
            x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        except:
            data = model.split_edges(data,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
            x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        try:
            pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
        except:
            print("Failed in Global Test")
            args.fail_counter += 1
            continue


        if pos_edge_index.size()[1] == 0 or neg_edge_index.size()[1] == 0:
            args.fail_counter += 1
            print("Failed on Graph")
            continue

        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index,weights)
        auc, ap = model.test(z, pos_edge_index, neg_edge_index)
        auc_list.append(auc)
        ap_list.append(ap)
    return auc_list, ap_list

def filter_dataset(dataset,min_nodes,max_nodes):
    filtered_dataset = []
    for i, graph in enumerate(dataset):
        num_nodes = graph.num_nodes
        if num_nodes >= min_nodes and num_nodes <= max_nodes:
            filtered_dataset.append(graph)
    return filtered_dataset

def run_analysis(args, meta_model, train_loader):
    degree_scores = avg_node_deg(args, meta_model, train_loader)
    print("Finished Computing Avg node degree difference")
    diff_num_nodes_scores = diff_num_nodes(args, meta_model, train_loader)
    print("Finished Computing node number difference")
    diff_num_edges_scores = diff_num_edges(args, meta_model, train_loader)
    print("Finished Computing edge number difference")
    diff_avg_clustering_coeff_scores = diff_avg_clustering_coeff(args, meta_model, train_loader)
    print("Finished Computing Avg clustering coefficient difference")
    diff_avg_triangles_scores = diff_avg_triangles(args, meta_model, train_loader)
    print("Finished Computing Avg triangles difference")
    emb_scores = avg_emb_cosine_mat(args, train_loader)
    print("Finished Computing Embedding Scores")
    sig_scores = compute_sig_graph_sim(args, meta_model, train_loader)
    print("Finished Sig Scores")
    wl_scores, wl_scores_neg = run_wl_kernel(args,train_loader,meta_model,\
            args.meta_train_edge_ratio)
    print("Finished WL Scores")
    try:
        deg_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
                degree_scores.reshape(-1,1))
        deg_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
                degree_scores.reshape(-1))
        print("Deg Spearman score %f p-value %f"%(deg_spearman_score[0],deg_spearman_score[1]))
        print("Deg Pearson score %f p-value %f"%(deg_pearson_score[0],deg_pearson_score[1]))
    except:
        print("Failed Degree Sim Test")

    emb_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            emb_scores.cpu().numpy().reshape(-1,1))
    emb_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            emb_scores.cpu().numpy().reshape(-1))

    spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            wl_scores.reshape(-1,1))
    pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            wl_scores.reshape(-1))

    spearman_score_neg = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            wl_scores_neg.reshape(-1,1))
    pearson_score_neg = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            wl_scores_neg.reshape(-1))

    diff_num_nodes_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1,1),\
            diff_num_nodes_scores.reshape(-1,1))
    diff_num_nodes_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1),\
            diff_num_nodes_scores.reshape(-1))

    diff_num_edges_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1, 1), \
            diff_num_edges_scores.reshape(-1, 1))
    diff_num_edges_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1), \
            diff_num_edges_scores.reshape(-1))

    diff_avg_clustering_coeff_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1, 1), \
            diff_avg_clustering_coeff_scores.reshape(-1, 1))
    diff_avg_clustering_coeff_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1), \
            diff_avg_clustering_coeff_scores.reshape(-1))

    diff_avg_triangles_spearman_score = spearmanr(sig_scores.cpu().numpy().reshape(-1, 1), \
            diff_avg_triangles_scores.reshape(-1, 1))
    diff_avg_triangles_pearson_score = pearsonr(sig_scores.cpu().numpy().reshape(-1), \
            diff_avg_triangles_scores.reshape(-1))

    print("Emb Spearman score %f p-value %f"%(emb_spearman_score[0],emb_spearman_score[1]))
    print("Emb Pearson score %f p-value %f"%(emb_pearson_score[0],emb_pearson_score[1]))
    print("Pos Spearman score %f p-value %f"%(spearman_score[0],spearman_score[1]))
    print("Pos Pearson score %f p-value %f"%(pearson_score[0],pearson_score[1]))
    print("Neg Spearman score %f p-value %f"%(spearman_score_neg[0],spearman_score_neg[1]))
    print("Neg Pearson score %f p-value %f"%(pearson_score_neg[0],pearson_score_neg[1]))
    print("Diff Num Nodes Spearman score %f p-value %f" % \
            (diff_num_nodes_spearman_score[0], diff_num_nodes_spearman_score[1]))
    print("Diff Num Nodes Pearson score %f p-value %f" % \
            (diff_num_nodes_pearson_score[0], diff_num_nodes_pearson_score[1]))
    print("Diff Num Edges Spearman score %f p-value %f" % \
            (diff_num_edges_spearman_score[0], diff_num_edges_spearman_score[1]))
    print("Diff Num Edges Pearson score %f p-value %f" % \
            (diff_num_edges_pearson_score[0], diff_num_edges_pearson_score[1]))
    print("Diff Avg Clustering Coeff Spearman score %f p-value %f" % \
            (diff_avg_clustering_coeff_spearman_score[0], diff_avg_clustering_coeff_spearman_score[1]))
    print("Diff Avg Clustering Coeff Pearson score %f p-value %f" % \
            (diff_avg_clustering_coeff_pearson_score[0], diff_avg_clustering_coeff_pearson_score[1]))
    print("Diff Avg Triangles Spearman score %f p-value %f" % \
            (diff_avg_triangles_spearman_score[0], diff_avg_triangles_spearman_score[1]))
    print("Diff Avg Triangles Pearson score %f p-value %f" % \
            (diff_avg_triangles_pearson_score[0], diff_avg_triangles_pearson_score[1]))

def calculate_max_nodes_in_dataset(dataset,min_nodes):
    max_nodes = 0
    graph_id = 0
    total_nodes = 0
    big_node_graphs = 0
    total_edges = 0
    for i, graph in enumerate(dataset):
        num_nodes = graph.num_nodes
        if num_nodes >= min_nodes:
            total_nodes += num_nodes
            total_edges += graph.edge_index.shape[1]
            big_node_graphs += 1
        if num_nodes > max_nodes:
            max_nodes = num_nodes
            graph_id = i
    avg_nodes = total_nodes / len(dataset)
    avg_edges = total_edges / len(dataset)
    print("Max nodes is %d in graph %d" %(max_nodes,graph_id))
    print("Avg nodes is %d" %(avg_nodes))
    print("Avg Edges is %d" %(avg_edges))
    print("Num 1000 nodes is %d| Total Graphs %d" %(big_node_graphs, len(dataset)))
    return max_nodes

def calc_adamic_adar_score(G, pos_edge_index, neg_edge_index):
    pos_list = pos_edge_index.t().detach().cpu().tolist()
    neg_list = neg_edge_index.t().detach().cpu().tolist()
    pos_preds, neg_preds = [],[]
    for pos_edge in pos_list:
        try:
            pos_score = list(nx.adamic_adar_index(G,[tuple(pos_edge)]))[0][2]
        except:
            pos_score = 0
        pos_preds.append(pos_score)

    for neg_edge in neg_list:
        try:
            neg_score = list(nx.adamic_adar_index(G,[tuple(neg_edge)]))[0][2]
        except:
            neg_score = 0
        neg_preds.append(neg_score)
    preds = list(pos_preds) + list(neg_preds)
    pos_y = pos_edge_index.new_ones(pos_edge_index.size(1))
    neg_y = neg_edge_index.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0).detach().cpu().numpy()
    return roc_auc_score(y, preds), average_precision_score(y, preds)

def calc_deepwalk_score(pos_edge_index, neg_edge_index,node_vectors,entity2index):
    '''

    :param G:
    :param pos_edge_index:
    :param neg_edge_index:
    :param node_vectors: node vectors from deepwalk model
    :return:
    '''
    pos_list = pos_edge_index.t().detach().cpu().tolist()
    neg_list = neg_edge_index.t().detach().cpu().tolist()
    pos_preds, neg_preds = [], []
    for pos_edge in pos_list:
        node1, node2= entity2index[pos_edge[0]], entity2index[pos_edge[1]]
        pos_score = cosine_similarity(node_vectors[node1].reshape(1,-1),\
                node_vectors[node2].reshape(1,-1))[0]
        # Scaling the cosine similarity between 0 to 1

        pos_preds.append(pos_score)

    for neg_edge in neg_list:
        node1, node2 = entity2index[neg_edge[0]], entity2index[neg_edge[1]]
        neg_score = cosine_similarity(node_vectors[node1].reshape(1,-1),\
                node_vectors[node2].reshape(1,-1))[0]
        # Scaling the cosine similarity between 0 to 1

        neg_preds.append(neg_score)
    preds = list(pos_preds) + list(neg_preds)
    pos_y = pos_edge_index.new_ones(pos_edge_index.size(1))
    neg_y = neg_edge_index.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0).detach().cpu().numpy()
    return roc_auc_score(y, preds), average_precision_score(y, preds)

def do_random_walks(G,path_length, alpha=0, rand=random.Random(), start=None):

  if start!= None:
      path=[start]
  else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes()))]

  while len(path) < path_length:
    cur = path[-1]
    num_neighbors= [n for n in G.neighbors(cur)]
    if len(num_neighbors) > 0:
        if rand.random() >= alpha:
            path.append(rand.choice(num_neighbors))
        else:
            path.append(path[0])
    else:
        break
  return [str(node) for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(do_random_walks(G,path_length, rand=rand, alpha=alpha, start=node))

    return walks

def train_deepwalk_model(G,number_walks = 10, walk_length = 80,seed= 0,epochs =1):
    '''
    Trains the deepwalk model for baseline puporses on the test set.
    :param edge_array:
    :return:
    '''
    # Random Walk Generation
    walks = build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    print("Training...")
    model = Word2Vec(walks, size=64, window=5, min_count=0, sg=1, hs=1, workers=5,iter=epochs)
    node_vectors = model.wv.vectors
    index2entity = model.wv.index2entity
    entity2index = {}
    for i in range(len(index2entity)):
        entity2index[int(index2entity[i])] = i
    return node_vectors,entity2index,index2entity


def create_nx_graph(data):
    edges_train = data.train_pos_edge_index.t().detach().cpu().tolist()
    G_train = nx.Graph()
    G_train.add_edges_from(edges_train)
    return G_train

def create_nx_graph_deepwalk(data):
    edges_test = data.test_pos_edge_index.t().detach().cpu().tolist()
    edges_train = data.train_pos_edge_index.t().detach().cpu().tolist()
    edges_val = data.val_pos_edge_index.t().detach().cpu().tolist()
    all_pos_edges = edges_train + edges_test + edges_val
    #Getting the test negative edges as we will evaluate deepwalk on that.
    edges_test_neg = data.test_neg_edge_index.t().detach().cpu().tolist()
    all_neg_edges_arr = np.array(edges_test_neg)

    G = nx.Graph()
    edges_arr= np.array(all_pos_edges)
    # Taking care of isolated nodes in the graph
    max_node_id = max(max(edges_arr[:,0]), max(edges_arr[:,1]))
    # Taking the max node index of test negative edges if present
    max_node_id = max(max_node_id, max(all_neg_edges_arr[:,0]), max(all_neg_edges_arr[:,1]))
    #Add nodes here
    G.add_nodes_from([x for x in range(max_node_id+1)])
    G.add_edges_from(edges_train)
    return G

def subsample_edges(G, num_edges):
    total_edges = G.edge_index.shape()[1]
    perm = torch.randperm(G.edge_index)
    subsampled_edges = perm[:num_edges]
    G.edge_index = subsampled_edges
    return G

def val(model, args, x, val_pos_edge_index, num_nodes, weights):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, val_pos_edge_index, weights)
        loss = model.recon_loss(z, val_pos_edge_index)
        if args.model in ['VGAE']:
            loss = loss + (1 / num_nodes) * model.kl_loss()
    return loss.item()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def create_masked_networkx_graph(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph`.
    Args:
        data (torch_geometric.data.Data): The data object.
    """

    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))

    values = {key: data[key].squeeze().tolist() for key in data.keys}

    for i, (u, v) in enumerate(data.train_pos_edge_index.t().tolist()):
        G.add_edge(u, v)
    return G

def avg_node_deg(args, model, dataloader):
    dataset = dataloader.dataset
    node_degree_list = []
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    for data_graph in dataset:
        sum_node_degree = 0
        data = model.split_edges(data_graph,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        graph_degree_list = []
        for node_id in nx_graph.nodes:
            sum_node_degree += nx_graph.degree[node_id]
            graph_degree_list.append(nx_graph.degree[node_id])
        log_median_node_degree = np.log(median(graph_degree_list))
        log_avg_node_degree = np.log(sum_node_degree / len(nx_graph))
        node_degree_list.append(log_median_node_degree)

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(node_degree_list[i] - node_degree_list[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_num_nodes(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    num_nodes_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev), \
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        num_nodes_per_dataset.append(len(nx_graph.nodes))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(num_nodes_per_dataset[i] - num_nodes_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_num_edges(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    num_edges_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        num_edges_per_dataset.append(len(nx_graph.edges))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(num_edges_per_dataset[i] - num_edges_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_avg_clustering_coeff(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    avg_clustering_coeff_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        avg_clustering_coeff_per_dataset.append(nx.average_clustering(nx_graph))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = np.absolute(avg_clustering_coeff_per_dataset[i] - avg_clustering_coeff_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def diff_avg_triangles(args, model, dataloader):
    dataset = dataloader.dataset
    meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
    scores_mat = np.zeros((len(dataset),len(dataset)))
    avg_triangles_per_dataset = []
    for data_graph in dataset:
        sum_node_degree = 0
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,
                                     val_ratio=args.meta_val_edge_ratio,
                                     test_ratio=meta_test_edge_ratio)
        nx_graph = create_masked_networkx_graph(data)
        avg_triangles_per_dataset.append(mean(nx.triangles(nx_graph).values()))

    for i in range(0,len(dataset)):
        for j in range(0,len(dataset)):
            score = abs(avg_triangles_per_dataset[i] - avg_triangles_per_dataset[j])
            scores_mat[i][j] = score

    return scores_mat

def avg_emb_cosine_mat(args, dataloader):
    dataset = dataloader.dataset
    embedding_list = []
    for data_graph in dataset:
        x = data_graph.x.to(args.dev)
        avg_emb = torch.mean(x,dim=0)
        norm = avg_emb.norm(p=2, dim=0, keepdim=True)
        norm_avg_emb = avg_emb.div(norm.expand_as(avg_emb))
        embedding_list.append(norm_avg_emb)
    avg_emb_mat = torch.stack(embedding_list)
    scores_mat = torch.mm(avg_emb_mat, avg_emb_mat.t())
    print("Avg Node Feature similarity between graphs: %f"%(scores_mat.mean()))
    return scores_mat

def compute_sig_graph_sim(args, model, dataloader):
    dataset = dataloader.dataset
    sig_params_list = []
    for data_graph in dataset:
        data_graph.train_mask = data_graph.val_mask = data_graph.test_mask = data_graph.y = None
        data_graph.batch = None
        num_nodes = data_graph.num_nodes
        # Val Ratio is Fixed at 0.2
        if args.concat_fixed_feats:
            if data_graph.x.shape[1] < args.num_features:
                concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                data_graph.x = torch.cat((data_graph.x,concat_feats),1)
        meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
        data_graph.x.cuda()
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev),\
                data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data = model.split_edges(data_graph,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)

        x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        test_pos_edge_index, test_neg_edge_index = data.test_pos_edge_index.to(args.dev),\
                data.test_neg_edge_index.to(args.dev)
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index, \
                    OrderedDict(model.named_parameters()), inner_loop=True)
        sig_out = model.encoder.cache_sig_out
        flattened_params = torch.cat(sig_out)
        norm = flattened_params.norm(p=2, dim=0, keepdim=True)
        norm_sig_out = flattened_params.div(norm.expand_as(flattened_params))
        sig_params_list.append(norm_sig_out)
    sig_params_mat = torch.stack(sig_params_list)
    scores_mat = torch.mm(sig_params_mat, sig_params_mat.t())
    return scores_mat

def create_erdos_renyi_graph(num_nodes,edge_prob):
    graph = erdos_renyi_graph(num_nodes,edge_prob)
    return graph


def to_torch_sparse_tensor(values,device):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    dim = values.shape[0]
    indices = torch.from_numpy(
            np.vstack(np.arange(dim), np.arange(dim)).astype(np.int64))
    values = torch.from_numpy(values)
    shape = torch.Size((dim,dim))
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def get_positional_embedding(edge_index,n,dim):
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n))
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    x = eigen_decomposision(n, dim, adj, dim, 10)
    return x


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sp.linalg.eigen.arpack.ArpackError:

            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                print('error')
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = torch.nn.functional.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def compute_acc_unsupervised(emb, task, extra_info):
    """
    Compute the accuracy of prediction given the labels.
    """

    if task == 'node':
        emb = emb.cpu().numpy()

        train_labels = extra_info['train_idx'][1].cpu().numpy()
        test_labels = extra_info['test_idx'][2].cpu().numpy()
        lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
        lr.fit(emb[extra_info['train_idx'][0]], train_labels)

        pred = lr.predict_proba(emb)[ extra_info['test_idx'][1]]

        pred = scatter(torch.FloatTensor(pred),index= extra_info['test_idx'][0].cpu(), dim=0, reduce="mean")
        pred = np.argmax(pred.cpu().numpy(), axis=1)
        return skm.f1_score(test_labels, pred, average='micro')
    else:
        test_index_positive = extra_info["test_map"][1]
        test_index_negative = extra_info["test_neg_map"][1]
        value_pos = torch.sigmoid((emb[test_index_positive[0]] * emb[test_index_positive[1]]).sum(dim=1))
        pos_pred = scatter(value_pos,index=extra_info['test_map'][0], dim=0, reduce="mean")
        value_neg = torch.sigmoid((emb[test_index_negative[0]] * emb[test_index_negative[1]]).sum(dim=1))
        neg_pred = scatter(value_neg,index=extra_info['test_neg_map'][0], dim=0, reduce="mean")
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = emb.new_ones(pos_pred.size(0))
        neg_y = emb.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred)


def compute_acc_unsupervised(task, feats,train_labels = None,test_labels = None,labels = None, test_index_positive = None, test_index_negative = None):
    """
    Compute the accuracy of prediction given the labels.
    """

    if task == 'node':
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()

        lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)

        lr.fit(feats[train_labels], labels[train_labels])
        pred = lr.predict_proba(feats)[test_labels]
        pred = np.argmax(pred, axis=1)

        return skm.f1_score(labels[test_labels], pred, average='micro')
    else:

        pos_pred = torch.sigmoid((feats[test_index_positive[0]] * feats[test_index_positive[1]]).sum(dim=1))
        neg_pred = torch.sigmoid((feats[test_index_negative[0]] * feats[test_index_negative[1]]).sum(dim=1))

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred)


def compute_acc_unsupervised_isolate(task, text_feats,vision_feats,structure_feats,train_labels = None,test_labels = None,labels = None, test_index_positive = None, test_index_negative = None):
    """
    Compute the accuracy of prediction given the labels.
    """

    if task == 'node':
        text_feats = text_feats.cpu().numpy()
        vision_feats = vision_feats.cpu().numpy()
        structure_feats = structure_feats.cpu().numpy()
        labels = labels.cpu().numpy()


        if len(labels.shape)== 2: lr = OneVsRestClassifier(lm.LogisticRegression( max_iter=10000))
        else:lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)


        lr.fit(text_feats[train_labels], labels[train_labels])
        pred_text = lr.predict_proba(text_feats)[test_labels]

        lr.fit(vision_feats[train_labels], labels[train_labels])
        pred_vision = lr.predict_proba(vision_feats)[test_labels]

        lr.fit(structure_feats[train_labels], labels[train_labels])
        pred_structure = lr.predict_proba(structure_feats)[test_labels]
        pred = (pred_text+pred_vision+pred_structure)/3
        print(pred)

        if len(labels.shape)== 2:pred = (pred > 0.5).astype(int)
        else:pred = np.argmax(pred, axis=1)

        return skm.f1_score(labels[test_labels], pred, average='micro')
    else:

        pos_pred_text = torch.sigmoid((text_feats[test_index_positive[0]] * text_feats[test_index_positive[1]]).sum(dim=1))
        neg_pred_text = torch.sigmoid((text_feats[test_index_negative[0]] * text_feats[test_index_negative[1]]).sum(dim=1))

        pos_pred_vision = torch.sigmoid((vision_feats[test_index_positive[0]] * vision_feats[test_index_positive[1]]).sum(dim=1))
        neg_pred_vision = torch.sigmoid((vision_feats[test_index_negative[0]] * vision_feats[test_index_negative[1]]).sum(dim=1))

        pos_pred_structure = torch.sigmoid((structure_feats[test_index_positive[0]] * structure_feats[test_index_positive[1]]).sum(dim=1))
        neg_pred_structure = torch.sigmoid((structure_feats[test_index_negative[0]] * structure_feats[test_index_negative[1]]).sum(dim=1))


        pred = torch.cat([(pos_pred_text+pos_pred_vision+pos_pred_structure)/3, (neg_pred_text+neg_pred_vision+neg_pred_structure)/3], dim=0)
        pos_y = pos_pred_structure.new_ones(pos_pred_structure.size(0))
        neg_y = neg_pred_structure.new_zeros(neg_pred_structure.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred)


def to_torch_coo_tensor(edge_index,edge_attr,size):
    size = (size, size)

    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
    )
    adj = adj._coalesced_(True)

    return adj


def gen_ran_output( model):
    vice_model = deepcopy(model)

    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):

        adv_param.data = param.data + 0.3*torch.ones_like(param.data)*param.data.std().detach()
    return vice_model


def loss_cal(x, x_aug):
    T = 0.1
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1))

    loss = - torch.log(loss).mean()

    return loss


def loss_cal2(output_positive, output_negative,auxiliary):

    summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
    discriminator_summary = auxiliary(summary_emb).T
    positive_score = output_positive @ discriminator_summary
    negative_score = output_negative @ discriminator_summary
    loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                        torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
        negative_score, torch.zeros_like(negative_score))


    return loss


def loss_cal3(output_positive, output_negative,auxiliary):

    T = 0.1
    output_positive = auxiliary(output_positive)
    output_negative = auxiliary(output_negative)


    batch_size, _ = output_positive.size()
    x_abs = output_positive.norm(dim=1)
    x_aug_abs = output_negative.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij',output_positive, output_negative) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1))

    loss = - torch.log(loss).mean()

    return loss


if __name__ =='__main__':
    datasets = ['academic','product','yelp','reddit']
    cum = 0
    rows = []
    cols = []
    new_edges = []
    for dataset in datasets:
        dataset_str = '/home/lmk/LAMP-GNN/lamp_data/' + dataset + '/'
        adj_full = sp.load_npz(dataset_str + 'adj_full.npz')
        print(adj_full.shape)
        adj_full = adj_full.tocoo()
        rows.append(adj_full.row+cum)
        cols.append(adj_full.col+cum)

        new_edges.append(np.arange(cum,cum+adj_full.shape[0]))
        cum+=adj_full.shape[0]


    for i in range(len(datasets)):
        rows.append(np.array([cum+i]*len(new_edges[i])))
        cols.append(new_edges[i])
        cols.append(np.array([cum+i]*len(new_edges[i])))
        rows.append(new_edges[i])

    tmp = np.ones((len(datasets),len(datasets)))
    for i in range(len(datasets)):tmp[i,i] = 0
    edges = tmp.nonzero()
    rows.append(edges[0]+cum)
    cols.append(edges[1]+cum)

    from scipy.sparse import coo_matrix,save_npz
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    coo = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(cum+len(datasets), cum+len(datasets)), dtype=np.int)
    save_npz('/data/lmk/adj_gcope.npz',coo.tocsr())


def compute_wasserstein_distance(X, Y):
    import ot
    from sklearn.decomposition import PCA


    X = X.cpu().numpy() if hasattr(X, 'cpu') else X
    Y = Y.cpu().numpy() if hasattr(Y, 'cpu') else Y

    pca = PCA(n_components=64)
    pca.fit(X)
    X = pca.transform(X)

    pca.fit(Y)
    Y = pca.transform(Y)

    # X: [n_x, d], Y: [n_y, d] node features from two graph domains


    n, m = X.shape[0], Y.shape[0]
    a = np.ones((n,)) / n  # uniform distribution over nodes in X
    b = np.ones((m,)) / m  # uniform distribution over nodes in Y
    M = ot.dist(X, Y, metric='euclidean')  # cost matrix
    distance = ot.emd2(a, b, M)  # squared Wasserstein distance

    return distance


def linear_CKA_torch(X, Y,device):
    """
    Compute the linear CKA between feature matrices X and Y.
    X: [n_samples, d1]
    Y: [n_samples, d2]
    """
    def rbf_kernel(X, sigma=None):
        GX = X @ X.T
        X_square = torch.diag(GX)
        pairwise_dists = X_square.unsqueeze(1) - 2 * GX + X_square.unsqueeze(0)

        if sigma is None:
            # Median heuristic for sigma
            dists = pairwise_dists.flatten()
            sigma = torch.sqrt(torch.median(dists[dists > 0]) + 1e-6)

        K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
        return K

    def center_gram(K):
        n = K.size(0)
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H


    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    K = rbf_kernel(X)
    L = rbf_kernel(Y)

    K = X @ X.T
    L = Y @ Y.T

    K_centered = center_gram(K)
    L_centered = center_gram(L)

    hsic = torch.trace(K_centered @ L_centered)
    var1 = torch.sqrt(torch.trace(K_centered @ K_centered))
    var2 = torch.sqrt(torch.trace(L_centered @ L_centered))

    return hsic / (var1 * var2 + 1e-8)


def random_mask( modal_ratio,N):
    modal1_list = np.random.choice(N,int(modal_ratio*N),replace = False)
    modal2_list = np.random.choice(N,int(modal_ratio*N),replace = False)
    modal3_list = np.random.choice(N,int(modal_ratio*N),replace = False)

    random_mask = torch.zeros(N,3)

    random_mask[modal1_list,0]=1
    random_mask[modal2_list,1]=1
    random_mask[modal3_list,2]=1


    return random_mask


def modal_random_mask(modal_ratio, N):
    # modal_ratio specifies the observation retention ratio.
    return make_modality_mask(N, 3, 1-modal_ratio,
                              seed=int(np.random.randint(0, 2**31-1)))


def complete_modalities_graph(mask ,text_feats, vision_feats, structure_feats):
    """
    Complete missing modalities using vectorized operations.
    text_feats, vision_feats, structure_feats: (N, d) torch.FloatTensor
    text_list, vision_list, structure_list: list[int]
    Returns: Completed text, vision, and structure features.
    """
    N, d = text_feats.shape


    feats = torch.stack([text_feats, vision_feats, structure_feats], dim=1)

    # Sum observed modality features for each node, shape (N, 1, d).
    sum_feats = feats * mask.unsqueeze(-1)          # (N, 3, d), retaining only observed modalities.
    sum_feats = sum_feats.sum(dim=1, keepdim=True)  # (N,1,d)

    # Count observed modalities per node, shape (N, 1, 1).
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)


    # Compute the mean of observed modality features, shape (N, 1, d).
    mean_feats = sum_feats / count

    # Use the mean of observed features to fill missing modalities.
    mask_expand = mask.unsqueeze(-1)  # (N,3,1)
    feats_completed = feats * mask_expand + mean_feats * (1 - mask_expand)


    text_feats_new, vision_feats_new, structure_feats_new = feats_completed.unbind(dim=1)


    return [text_feats_new,vision_feats_new,structure_feats_new]


def complete_modalities_graph2(mask ,text_feats, vision_feats, structure_feats):
    """
    Complete missing modalities using vectorized operations.
    text_feats, vision_feats, structure_feats: (N, d) torch.FloatTensor
    text_list, vision_list, structure_list: list[int]
    Returns: Completed text, vision, and structure features.
    """
    N, d = text_feats.shape


    feats = torch.stack([text_feats, vision_feats, structure_feats], dim=1)

    # Sum observed modality features for each node, shape (N, 1, d).
    sum_feats = feats * mask.unsqueeze(-1)          # (N, 3, d), retaining only observed modalities.
    sum_feats = sum_feats.sum(dim=1, keepdim=True)  # (N,1,d)

    # Count observed modalities per node, shape (N, 1, 1).
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)


    # Compute the mean of observed modality features, shape (N, 1, d).
    mean_feats = (sum_feats / count).squeeze(1)

    # Use the mean of observed features to fill missing modalities.


    return [mean_feats,mean_feats,mean_feats]


def get_index(mask):
    """Return the missing-node indices for each modality."""
    return [torch.nonzero(mask[:, m] == 0, as_tuple=True)[0]
            for m in range(mask.size(1))]


def gumbel_topk(logits, k, tau=0.5, training=True, return_mask=False):
    """Straight-through Gumbel top-K selection, Eq. (10).

    Training uses Gumbel perturbations and progressive soft exclusion.
    Evaluation uses deterministic top-K selection. Aggregation weights are
    normalized from the unperturbed relevance logits, Eq. (11).
    """
    if not 1 <= k <= logits.size(-1) or tau <= 0:
        raise ValueError('Require 1 <= k <= factor count and tau > 0')
    scores = logits
    if training:
        eps = torch.finfo(logits.dtype).eps
        u = torch.rand_like(logits).clamp(eps, 1-eps)
        scores = logits - torch.log(-torch.log(u))
    indices = scores.topk(k, dim=-1).indices
    hard = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
    if training:
        running = scores
        soft = torch.zeros_like(logits)
        for _ in range(k):
            prob = torch.softmax(running / tau, dim=-1)
            soft = soft + prob
            running = running + torch.log((1-prob).clamp_min(1e-7))
        selection = hard + (soft-soft.detach())
    else:
        selection = hard
    weights = selected_softmax(logits, selection)
    return (weights, indices, selection) if return_mask else (weights, indices)


def selected_softmax(logits, selection):
    """Normalize exp(logits) over selected entries, preserving ST gradients.

    Also used on flattened modality/factor entries for Eq. (12), so duplicate
    factor selections from different modalities remain distinct entries.
    """
    active = selection.detach() > 0
    shift = logits.detach().masked_fill(~active, -float('inf')).amax(-1, keepdim=True)
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    mass = torch.exp((logits-shift).clamp(-80, 80)) * selection
    return mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


def semantic_grounding_loss(query, factors, weights, observed=None, eta=0.25):
    """Semantic grounding loss with L2 distances and stop-gradient, Eq. (16).

    Loss terms are summed over participating nodes and selected factors.
    Retrieval weights remain differentiable.
    """
    loss = weights * (torch.cdist(query.detach(), factors, p=2)
                      + eta * torch.cdist(query, factors.detach(), p=2))
    if observed is not None:
        loss = loss * observed[:, None]
    return loss.sum()


def make_modality_mask(num_nodes, num_modalities=3, missing_rate=0.6,
                       mode='random', edge_index=None, seed=None, max_hop=2):
    """Appendix A.3. CPU masks with an explicit, feasible observation budget.

    Random: balanced one-per-node assignment, then random additions to nearly
    equal per-modality counts. Degree: remove low-degree nodes first. Cluster:
    remove a common modality in each snowball group. Every node retains >=1.
    """
    from collections import deque
    if num_nodes < 1 or num_modalities < 1:
        raise ValueError('Empty graph/modalities')
    if not 0 <= missing_rate <= 1 - 1/num_modalities + 1e-9:
        raise ValueError('Missing rate is incompatible with >=1 modality/node')
    rng = np.random.default_rng(seed)
    n, m = num_nodes, num_modalities
    keep = max(n, min(n*m, int(round(n*m*(1-missing_rate)))))
    if mode == 'random':
        mask = np.zeros((n, m), dtype=np.float32)
        order = rng.permutation(n)
        mod_order = rng.permutation(m)
        assigned = mod_order[np.arange(n) % m]
        mask[order, assigned] = 1
        counts = mask.sum(0).astype(int)
        targets = np.full(m, keep//m, dtype=int)
        # Extra budget goes to the already larger columns first (ties random).
        priority = np.lexsort((rng.random(m), -counts))
        targets[priority[:keep % m]] += 1
        for col in range(m):
            candidates = np.flatnonzero(mask[:, col] == 0)
            chosen = rng.choice(candidates, targets[col]-counts[col], replace=False)
            mask[chosen, col] = 1
        return torch.from_numpy(mask)
    if mode not in ('degree', 'cluster', 'snowball'):
        raise ValueError('Unknown missingness mechanism: '+mode)
    if edge_index is None:
        raise ValueError('Structural missingness requires edge_index')
    edges = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    neighbors = [set() for _ in range(n)]
    for u, v in edges.T:
        if u != v:
            neighbors[int(u)].add(int(v)); neighbors[int(v)].add(int(u))
    mask = np.ones((n, m), dtype=np.float32)
    remaining = n*m-keep
    if mode == 'degree':
        order = np.lexsort((rng.random(n), np.array([len(v) for v in neighbors])))
        for node in order:
            for col in rng.permutation(m):
                if remaining == 0:
                    return torch.from_numpy(mask)
                if mask[node].sum() > 1:
                    mask[node, col] = 0
                    remaining -= 1
    else:
        for _ in range(m):
            for start in rng.permutation(n):
                if remaining == 0:
                    return torch.from_numpy(mask)
                group, seen = [], {int(start)}
                queue = deque([(int(start), 0)])
                while queue:
                    node, depth = queue.popleft()
                    group.append(node)
                    if depth < max_hop:
                        for nxt in sorted(neighbors[node]):
                            if nxt not in seen:
                                seen.add(nxt); queue.append((nxt, depth+1))
                for col in rng.permutation(m):
                    for node in group:
                        if remaining == 0:
                            return torch.from_numpy(mask)
                        if mask[node, col] and mask[node].sum() > 1:
                            mask[node, col] = 0
                            remaining -= 1
                    # Keep the local group correlated for this expansion.
                    break
        # Complete the observation budget across remaining nodes.
        for node in rng.permutation(n):
            for col in rng.permutation(m):
                if remaining == 0:
                    return torch.from_numpy(mask)
                if mask[node, col] and mask[node].sum() > 1:
                    mask[node, col] = 0; remaining -= 1
    return torch.from_numpy(mask)

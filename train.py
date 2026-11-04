from copy import deepcopy
from unittest import result
import torch
import torch.nn.functional as F
import sklearn.neighbors
import copy
from collections import OrderedDict
import numpy as np
from utils import test,test2,seed_everything
from models.models import MLP,WDiscriminator,WDiscriminator_old,matchGAT
from models.autoencoder import negative_sampling
import time
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import coo_array
from data.load_data import *
from utils import *


EPS = 1e-15

def construct_space(model,edges_num,train_pos_edge_index,graph_x):

    if model is None:z = graph_x
    else:z = model(graph_x)

    node_embs = torch.hstack((z, z))


    if train_pos_edge_index.shape[1] <= edges_num:
        select_edges = train_pos_edge_index
    else:
        perm = np.random.choice(train_pos_edge_index.shape[1], edges_num, replace=False)
        select_edges = train_pos_edge_index[:, perm]

    edge_embs = torch.hstack((z[select_edges[0]], z[select_edges[1]]))
    source_space = torch.vstack((node_embs, edge_embs))
    return source_space


def train_wdiscriminator_old(space_s, space_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True


    for j in range(batch_d_per_iter):
        wdiscriminator.train()


        w0 = wdiscriminator(space_s)
        w1 = wdiscriminator(space_t)


        loss = -torch.mean(w1) + torch.mean(w0)


        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)


    return wdiscriminator


def compute_wdiscriminator_by_nn(graph_s, graph_t,device, batch_d_per_iter=30):
    wdiscriminator = WDiscriminator(128).to(device)
    optimizer_d = torch.optim.Adam(wdiscriminator.parameters(), lr=0.01, weight_decay=5e-4)
    wdiscriminator.reset_parameters()
    for p in wdiscriminator.parameters(): p.requires_grad = True
    if not isinstance(graph_t,list):graph_t = [graph_t]
    for j in range(batch_d_per_iter):
        wdiscriminator.train()
        w1s = []
        for graph in graph_t:
            w1s.append(wdiscriminator(graph.x.detach(),graph.edge_index))
        w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index)
        w1 = torch.vstack(w1s)

        loss = -torch.mean(w1) + torch.mean(w0)


        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

    return -loss.item()


def train_wdiscriminator(graph_s, graph_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True


    if not isinstance(graph_t,list):graph_t = [graph_t]


    for j in range(batch_d_per_iter):
        wdiscriminator.train()

        w1s = []
        for graph in graph_t:
            w1s.append(wdiscriminator(graph.x.detach(),graph.edge_index))


        w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index)
        w1 = torch.vstack(w1s)

        loss = -torch.mean(w1) + torch.mean(w0)


        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

    return wdiscriminator


def train_wdiscriminator_align(graph_s, graph_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True


    for j in range(batch_d_per_iter):
        wdiscriminator.train()


        w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index)
        w1 = wdiscriminator(graph_t.x.detach(),graph_t.edge_index)

        loss = -torch.mean(w1) + torch.mean(w0)


        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)


    w0 = wdiscriminator(graph_s.x.detach(),graph_s.edge_index).cpu().detach().numpy()[:,0]
    w1 = wdiscriminator(graph_t.x.detach(),graph_t.edge_index).cpu().detach().numpy()[:,0]

    index0 = np.argsort(w0)
    index1 = np.argsort(-w1)


    return index0[np.argsort(index1)]


def domain_tansformation_old(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    wdiscriminator = WDiscriminator_old(args.share_dims*2).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)
        for graph_t in graph_syn.syn_graphs:

            space_s = construct_space(model,graph_s.x.size(0)*2,graph_s.edge_index,graph_s.x)
            space_t = construct_space(None,graph_t.x.size(0)*2,graph_t.edge_index,graph_t.x)
            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator_old(space_s.detach(), space_t.detach(),graph_s.x.size(0), wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w0 = wdiscriminator_copy(space_s[:graph_s.x.size(0)])
            w1 = wdiscriminator_copy(space_t[:graph_t.x.size(0)])

            w2 = wdiscriminator_copy(space_s[graph_s.x.size(0):])
            w3 = wdiscriminator_copy(space_t[graph_t.x.size(0):])

            loss = (torch.mean(w1) - torch.mean(w0) + torch.mean(w3) - torch.mean(w2))

            loss_all+=loss


        value = loss_all.item()

        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1
        print(cnt)
        if cnt >0:return best_model


        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def domain_tansformation(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)
        for graph_t in graph_syn.syn_graphs:

            wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_t,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False

            wdiscriminator_copy.to(device)


            w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)
            w1 = wdiscriminator_copy(graph_t.x, graph_t.edge_index)

            loss = torch.mean(w1) - torch.mean(w0)


            loss_all+=loss


        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value)
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1


        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def domain_tansformation2(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)

        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_syn.syn_graphs,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=40))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False

        wdiscriminator_copy.to(device)


        w1s = []
        w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)

        for graph in graph_syn.syn_graphs:
            w1s.append(wdiscriminator_copy(graph.x, graph.edge_index))
        w1 = torch.vstack(w1s)
        loss_all = torch.mean(w1) - torch.mean(w0)


        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value)
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1


        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def domain_tansformation3(graph_s,graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    model_out = MLP(args.share_dims, args.hidden_dims, graph_s.x.shape[1]).to(device)


    optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.001},{'params':model_out.parameters(),'lr':0.0005}])

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)

        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_syn.syn_graphs,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=80))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False

        wdiscriminator_copy.to(device)


        w1s = []
        w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)

        for graph in graph_syn.syn_graphs:
            w1s.append(wdiscriminator_copy(graph.x, graph.edge_index))
        w1 = torch.vstack(w1s)
        loss = torch.mean(w1) - torch.mean(w0)

        loss_AE =torch.nn.MSELoss(size_average=True)(graph_s.x,model_out(model(graph_s.x)))

        loss_all = loss_AE


        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value,loss.item(),loss_AE.item())
        if cnt >2:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1
        print(cnt)


        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def autoencoder(args,dataset,device,num):
    dataset_str = '/home/lmk/LAMP-GNN/lamp_data/' + dataset + '/'
    feat = np.load(dataset_str + 'features.npy',mmap_mode='r')
    model = MLP(feat.shape[1], args.hidden_dims, 1130).to(device)
    model_out = MLP(1130, args.hidden_dims, feat.shape[1]).to(device)
    optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.0001},{'params':model_out.parameters(),'lr':0.0001}])
    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:
        model.train()
        model_out.train()
        index = np.random.choice(feat.shape[0],100000,replace=False)
        features = torch.FloatTensor(feat[index,:]).to(device)


        loss_AE =torch.nn.MSELoss(size_average=True)(features,model_out(model(features)))

        value = loss_AE.item()

        if cnt >100:
            print('XXXXXXXXXXXXXXXXXXXXX',value,cnt)
            return torch.save(best_model.state_dict(),'./ood/autoencoder_for_{}_{}.pth'.format(dataset,num))
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1


        optimizer.zero_grad()
        loss_AE.backward()
        optimizer.step()


def compute_group_wasserstein(args,device):
    class domain:
        def __init__(self,ds,subgraph_scale = 4096):
            super(domain, self).__init__()
            self.train_dataset = ds
            self.test_dataset = [None]
            self.subgraph_scale = subgraph_scale

    datasets = ['academic','product','reddit','yelp']
    dimension = {'academic':128,'product':100,'yelp':300,'reddit':602}
    if os.path.exists('./autoencoder_mlp.pth'):
        mlps = {}
        for dataset in datasets:
            mlp = MLP(dimension[dataset], 128, 128).to(device)
            mlp.load_state_dict(torch.load('./autoencoder_mlp.pth')[dataset], strict=True)
            mlps[dataset] = mlp
    else:
        tmp = {}
        mlps = {}
        for dataset in datasets:
            dataset_str = '/home/lmk/LAMP-GNN/lamp_data/' + dataset + '/'
            feat = np.load(dataset_str + 'features.npy',mmap_mode='r')

            scaler = StandardScaler()
            scaler.fit(feat)
            feat = scaler.transform(feat)

            model = MLP(feat.shape[1], 128, 128).to(device)
            model_out = MLP( 128, 128, feat.shape[1]).to(device)
            optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.0001},{'params':model_out.parameters(),'lr':0.0001}])
            best_value = 1e8
            cnt = 0
            best_model = copy.deepcopy(model)
            while cnt<200:
                model.train()
                model_out.train()
                index = np.random.choice(feat.shape[0],10000,replace=False)
                features = torch.FloatTensor(feat[index,:]).to(device)
                loss_AE =torch.nn.MSELoss(size_average=True)(features,model_out(model(features)))
                value = loss_AE.item()

                if value < 0:
                    cnt+=1
                elif value>=0 and value < best_value:
                    best_value = value
                    cnt = 0
                    best_model = copy.deepcopy(model)
                else: cnt+=1
                print('XXXXXXXXXXXXXXXXXXXXX',value,cnt)
                optimizer.zero_grad()
                loss_AE.backward()
                optimizer.step()
            tmp[dataset] = best_model.state_dict()
            mlps[dataset] = best_model
        torch.save(tmp,'./autoencoder_mlp.pth')

    dataloader = dataGraph(domain(datasets,4096))
    subgraphs = dataloader.fetch_subgraph()
    for dataset in datasets:
        subgraphs[dataset].to_tensor(device)
        subgraphs[dataset].original_x = subgraphs[dataset].x
        subgraphs[dataset].x = mlps[dataset](subgraphs[dataset].original_x).detach()


    for i in range(4):
        for j in range(i+1,4):
            print(datasets[i],datasets[j])
            results = []
            for _ in range(10):results.append(compute_wdiscriminator_by_nn(subgraphs[datasets[i]], subgraphs[datasets[j]],device))

            print(np.mean(results))


def pretrain_autoencoder(args,dataset,device):
    dataset_str = '/home/lmk/LAMP-GNN/lamp_data/' + dataset + '/'
    feat = np.load(dataset_str + 'features.npy',mmap_mode='r')

    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)


    model = MLP(feat.shape[1], args.hidden_dims, args.share_dims).to(device)
    model_out = MLP( args.share_dims, args.hidden_dims, feat.shape[1]).to(device)
    optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.0001},{'params':model_out.parameters(),'lr':0.0001}])
    best_value = 1e8
    cnt = 0
    best_model = [copy.deepcopy(model),copy.deepcopy(model_out)]
    while True:
        model.train()
        model_out.train()
        index = np.random.choice(feat.shape[0],100000,replace=False)
        features = feat[index,:]
        features = torch.FloatTensor(features).to(device)


        loss_AE =torch.nn.MSELoss(reduction='mean' )(features,model_out(model(features)))

        value = loss_AE.item()

        if cnt >200:
            print('XXXXXXXXXXXXXXXXXXXXX',value,cnt)
            return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = [copy.deepcopy(model),copy.deepcopy(model_out)]
        else: cnt+=1


        optimizer.zero_grad()
        loss_AE.backward()
        optimizer.step()


def pretrain_autoencoder_wasserstein(args,datasets,device):

    def share_space(models,features):
        space = []
        for dataset in datasets:
            space.append(models[dataset][0](features[dataset]))
        return torch.cat(space,dim = 0).detach()

    def train_wdiscriminator_node(embedding_s, embedding_t, batch_d_per_iter=20):

        wdiscriminator = WDiscriminator_old(args.share_dims).to(device)
        optimizer_d = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

        wdiscriminator.reset_parameters()

        for p in wdiscriminator.parameters(): p.requires_grad = True
        wdiscriminator.train()

        for j in range(batch_d_per_iter):

            optimizer_d.zero_grad()
            w0 = wdiscriminator(embedding_s)
            w1 = wdiscriminator(embedding_t)

            l1 = -torch.mean(w1) + torch.mean(w0)
            loss = l1


            loss.backward()
            optimizer_d.step()

            for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

        loss = -torch.mean(w1) + torch.mean(w0)
        print(loss.item())

        return wdiscriminator


    feats = {}
    models = {}
    params = []

    for dataset in datasets:
        dataset_str = '/home/lmk/LAMP-GNN/lamp_data/' + dataset + '/'
        feat = np.load(dataset_str + 'features.npy',mmap_mode='r')

        scaler = StandardScaler()
        scaler.fit(feat)
        feat = scaler.transform(feat)

        feats[dataset] = feat
        model = MLP(feat.shape[1], args.hidden_dims, args.share_dims).to(device)
        model_out = MLP( args.share_dims, args.hidden_dims, feat.shape[1]).to(device)
        models[dataset] = (model,model_out)

        params.extend(model.parameters())
        params.extend(model_out.parameters())


    optimizer = torch.optim.Adam([{'params':params,'lr':0.001}]) 

    best_value = 1e8
    cnt = 0
    best_model = 0
    while True:


        features = {}
        for dataset in datasets:
            models[dataset][0].train()
            models[dataset][1].train()
            index = np.random.choice(feats[dataset].shape[0],10000,replace=False)
            features[dataset] = torch.FloatTensor(feats[dataset][index,:]).to(device)
        space = share_space(models,features)
        loss_Diss = []
        loss_AEs = []
        for dataset in datasets:
            embedding_s = models[dataset][0](features[dataset])
            wdiscriminator = train_wdiscriminator_node(embedding_s.detach(), space, batch_d_per_iter=20)
            for p in wdiscriminator.parameters(): p.requires_grad = False

            w0 = wdiscriminator(embedding_s)
            w1 = wdiscriminator(space)
            loss_Dis = torch.mean(w1) - torch.mean(w0)
            loss_AE =torch.nn.MSELoss(size_average=True)(features[dataset],models[dataset][1](models[dataset][0](features[dataset])))
            loss_Diss.append(loss_Dis)
            loss_AEs.append(loss_AE)

        loss_Dis = torch.stack(loss_Diss).mean()
        loss_AE = torch.stack(loss_AEs).mean()
        print(loss_Dis,loss_AE)
        loss = loss_Dis+loss_AE*10
        value = loss.item()

        if cnt >200:

            return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = models
        else: cnt+=1
        print('XXXXXXXXXXXXXXXXXXXXX',value,cnt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pretrain_graph_autoencoder_wasserstein(args,datasets,device):

    class domain:
        def __init__(self,ds,subgraph_scale = 4096):
            super(domain, self).__init__()
            self.train_dataset = ds
            self.test_dataset = [None]
            self.subgraph_scale = subgraph_scale


    dataloader = dataGraph(domain(datasets,args.subgraph_scale))


    models = {}
    params = []
    for dataset in datasets:

        model = MLP(args.data_dims[dataset], args.hidden_dims, args.share_dims).to(device)
        model_out = MLP( args.share_dims, args.hidden_dims, args.data_dims[dataset]).to(device)
        models[dataset] = (model,model_out)

        params.extend(model.parameters())
        params.extend(model_out.parameters())


    optimizer = torch.optim.Adam([{'params':params,'lr':0.001}])
    best_value = 1e8
    cnt = 0
    best_model = 0
    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)
    while True:
        loss_Diss = []
        loss_AEs = []
        subgraphs = dataloader.fetch_subgraph()
        for dataset in datasets:
            subgraphs[dataset].to_tensor(device)
            subgraphs[dataset].original_x = subgraphs[dataset].x
            subgraphs[dataset].x = models[dataset][0](subgraphs[dataset].original_x)

        for dataset in datasets:
            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(subgraphs[dataset], list(subgraphs.values()), wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=20))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)

            w1s = []
            for graph in subgraphs.values():
                w1s.append(wdiscriminator_copy(graph.x,graph.edge_index))
            w0 = wdiscriminator_copy(subgraphs[dataset].x,subgraphs[dataset].edge_index)
            w1 = torch.vstack(w1s)
            loss_Dis = torch.mean(w1) - w0
            loss_Diss.append(loss_Dis)

            loss_AE =torch.nn.MSELoss(size_average=True)(subgraphs[dataset].original_x,models[dataset][1](subgraphs[dataset].x))
            loss_AEs.append(loss_AE)


        loss_Dis = torch.stack(loss_Diss).mean()
        loss_AE = torch.stack(loss_AEs).mean()
        print(loss_Dis,loss_AE)
        loss = loss_Dis+loss_AE*10
        value = loss.item()

        if cnt >200:

            return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = models
        else: cnt+=1
        print('XXXXXXXXXXXXXXXXXXXXX',value,cnt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_graph_isolate(modality_projections,model, graph,optimizer,dims,device,file=None):

    for modal in graph.modal_names: modality_projections[modal].to(device)


    value = 0
    count = 0
    result = []

    for epoch in range(1,1500):


        model.train()
        for modal in graph.modal_names: modality_projections[modal].train()

        weights = OrderedDict(model.named_parameters())

        xs = []
        for modal in graph.modal_names:
            modal_feat = modality_projections[modal](graph.original_feats[modal])
            modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
            xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

        xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
        if model.task == 'link':
            x_text = model.encode(xs['text'], graph.train_edge_index, weights, inner_loop=True)
            x_vision = model.encode(xs['vision'], graph.train_edge_index, weights, inner_loop=True)
            x_structure = model.encode(xs['structure'], graph.train_edge_index, weights, inner_loop=True)


        else:
            x_text = model.encode(xs['text'], graph.edge_index, weights, inner_loop=True)
            x_vision = model.encode(xs['vision'], graph.edge_index, weights, inner_loop=True)
            x_structure = model.encode(xs['structure'], graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':
            loss = model.recon_loss_isolate(x_text,x_vision,x_structure,graph.train_edge_index)
        else:
            loss = model.class_loss_isolate(x_text[graph.train_labels],x_vision[graph.train_labels],x_structure[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        for modal in graph.modal_names: modality_projections[modal].eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            xs = []
            for modal in graph.modal_names:
                modal_feat = modality_projections[modal](graph.original_feats[modal])
                modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
                xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

            xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
            if model.task == 'link':
                x_text = model.encode(xs['text'], graph.train_edge_index, weights, inner_loop=True)
                x_vision = model.encode(xs['vision'], graph.train_edge_index, weights, inner_loop=True)
                x_structure = model.encode(xs['structure'], graph.train_edge_index, weights, inner_loop=True)
            else:
                x_text = model.encode(xs['text'], graph.edge_index, weights, inner_loop=True)
                x_vision = model.encode(xs['vision'], graph.edge_index, weights, inner_loop=True)
                x_structure = model.encode(xs['structure'], graph.edge_index, weights, inner_loop=True)


            if model.task == 'link':score = model.test_isolate(x_text,x_vision,x_structure,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else:

                score = model.class_test_isolate(x_text[graph.test_labels],x_vision[graph.test_labels],x_structure[graph.test_labels],graph.labels[graph.test_labels])


        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]:print(epoch,value,loss.item())


        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result


def train_graph_isolate_visualization(modality_projections,model, graph,optimizer,dims,device,file=None):

    for modal in graph.modal_names: modality_projections[modal].to(device)


    value = 0
    count = 0
    result = []

    for epoch in range(1,1500):


        model.train()
        for modal in graph.modal_names: modality_projections[modal].train()

        weights = OrderedDict(model.named_parameters())

        xs = []

        for modal in graph.modal_names:
            modal_feat = modality_projections[modal](graph.original_feats[modal])

            modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
            xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))


        xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
        if model.task == 'link':
            x_text = model.encode(xs['text'], graph.train_edge_index, weights, inner_loop=True)
            x_vision = model.encode(xs['vision'], graph.train_edge_index, weights, inner_loop=True)
            x_structure = model.encode(xs['structure'], graph.train_edge_index, weights, inner_loop=True)


        else:
            x_text = model.encode(xs['text'], graph.edge_index, weights, inner_loop=True)
            x_vision = model.encode(xs['vision'], graph.edge_index, weights, inner_loop=True)
            x_structure = model.encode(xs['structure'], graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':
            loss = model.recon_loss_isolate(x_text,x_vision,x_structure,graph.train_edge_index)
        else:
            loss = model.class_loss_isolate(x_text[graph.train_labels],x_vision[graph.train_labels],x_structure[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        for modal in graph.modal_names: modality_projections[modal].eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            xs = []
            xs_old = []
            for modal in graph.modal_names:
                modal_feat = modality_projections[modal](graph.original_feats[modal])
                modal_dim = int(np.sum(dims[:graph.modal_names.index(modal)]))
                xs.append(torch.cat([modal_feat[:,:modal_dim],graph.original_feats[modal],modal_feat[:,modal_dim:]],dim=1))

                modal_zero = torch.zeros_like(modal_feat)
                xs_old.append(torch.cat([graph.original_feats[modal],modal_zero],dim=1))


            xs = graph.complete_modalities_graph(xs[0],xs[1],xs[2])
            if model.task == 'link':
                x_text = model.encode(xs['text'], graph.train_edge_index, weights, inner_loop=True)
                x_vision = model.encode(xs['vision'], graph.train_edge_index, weights, inner_loop=True)
                x_structure = model.encode(xs['structure'], graph.train_edge_index, weights, inner_loop=True)
            else:
                x_text = model.encode(xs['text'], graph.edge_index, weights, inner_loop=True)
                x_vision = model.encode(xs['vision'], graph.edge_index, weights, inner_loop=True)
                x_structure = model.encode(xs['structure'], graph.edge_index, weights, inner_loop=True)


            if model.task == 'link':score = model.test_isolate(x_text,x_vision,x_structure,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else:
                score = model.class_test_isolate(x_text[graph.test_labels],x_vision[graph.test_labels],x_structure[graph.test_labels],graph.labels[graph.test_labels])


            if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]:print(epoch,value,loss.item())


            if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
            if value<score:


                embedding = (x_text[graph.test_labels]+x_vision[graph.test_labels]+ x_structure[graph.test_labels])/3
                labels = graph.labels[graph.test_labels]
                value = score
                count = 0
            else:count+=1
            result.append(score)

    if file!=None:file.write('the best value:,{}\n'.format(value))
    embedding = embedding.cpu().numpy()
    labels = labels.cpu().numpy()

    np.savez('./results/primg_booknc_embedding',embedding= embedding,labels=labels)


def calibrate_semantic_space(modality_projections, semantic_space, model, graph, optimizer, device, file=None,
                calibration_adapter=None, epochs=1000, patience=100, checkpoint_path=None):
    """Train the calibration adapter and task head, selecting by validation, Eq. (18)."""
    if calibration_adapter is None:
        raise ValueError('Semantic calibration requires a SemanticCalibrationAdapter')
    frozen = [model.encoder, semantic_space]+list(modality_projections.values())
    for module in frozen:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    expected = {id(p) for p in calibration_adapter.parameters()} | {id(p) for p in model.decoder.parameters()}
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    if expected != optimized:
        raise ValueError('Adaptation optimizer must contain only the calibration adapter and task head')
    best_value, best, count, best_epoch = -float('inf'), None, 0, 0
    for epoch in range(1, epochs+1):
        calibration_adapter.train(); model.decoder.train()
        # Propagate calibration gradients through the frozen encoder.
        embeddings, _ = encode_pass(model, semantic_space, modality_projections, graph, graph.mask,
                                     calibration_adapter=calibration_adapter, compute_grounding=False)
        embedding = torch.cat(embeddings, dim=1)
        if model.task == 'link':
            embedding = model.decoder.mlp(embedding)
            loss = model.recon_loss(embedding, graph.train_edge_index,
                                    neg_edge_index=graph.train_edge_index_negative)
        else:
            loss = model.class_loss(embedding[graph.train_labels], graph.labels[graph.train_labels])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        score = evaluate_graph(modality_projections, semantic_space, model, graph, calibration_adapter, split='val')
        if score > best_value:
            best_value, best_epoch, count = score, epoch, 0
            best = {'calibration_adapter': deepcopy(calibration_adapter.state_dict()),
                    'decoder': deepcopy(model.decoder.state_dict()),
                    'epoch': epoch, 'validation_score': score,
                    'split_digest': graph.split_digest, 'pretrain_id': graph.pretrain_id,
                    'adapter_rank': calibration_adapter.down.out_features}
        else:
            count += 1
        if file is not None:
            file.write(f'{epoch},{loss.item()},{score}\n'); file.flush()
        if patience > 0 and count >= patience:
            break
    if best is None:
        raise ValueError('No valid adaptation checkpoint; check epochs and labels')
    calibration_adapter.load_state_dict(best['calibration_adapter'])
    model.decoder.load_state_dict(best['decoder'])
    if checkpoint_path is not None:
        torch.save(best, checkpoint_path)
    # Test is evaluated once, after validation-based model selection.
    test_score = evaluate_graph(modality_projections, semantic_space, model, graph, calibration_adapter, split='test')
    return {'validation_score': best_value, 'test_score': test_score,
            'best_epoch': best_epoch, 'metric': 'auc' if model.task == 'link' else 'micro_f1'}


def calibrate_semantic_space_from_graph(modality_projections, semantic_space, model, graph, optimizer, device, file=None, **kwargs):

    return calibrate_semantic_space(modality_projections, semantic_space, model, graph, optimizer, device, file, **kwargs)


def train_graph_prompt(args,mlp,model, graph,optimizer,device,file=None):

    if mlp!= None:
        for data in args.train_dataset:mlp[data].to(device)
    model.to(device)

    value = 0
    count = 0
    result = []
    for epoch in range(1,1000):

        model.train()
        if mlp!= None:

            for data in args.train_dataset:mlp[data].train()

        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            xs = []
            for data in args.train_dataset:
                if data == args.test_dataset[0]:xs.append(graph.x)
                else: xs.append(mlp[data](graph.x))
            x = torch.cat(xs,dim = 1)
            if model.task == 'link':z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss(z[graph.train_labels], graph.labels[graph.train_labels])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        if mlp!= None:
            for data in args.train_dataset:mlp[data].eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())

            if mlp!= None:
                xs = []
                for data in args.train_dataset:
                    if data == args.test_dataset[0]:xs.append(graph.x)
                    else: xs.append(mlp[data](graph.x))
                x = torch.cat(xs,dim = 1)
                if model.task == 'link':z = model.encode(x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(x, graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else:
                score = model.class_test(z[graph.test_labels], graph.labels[graph.test_labels])


        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,value,loss.item())


        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result


def train_graph_by_link3(mlp,model, graph,optimizer,device,file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0

    if model.task != 'link':

        if len(graph.labels.shape) == 2:
            task_name = 'node_m'
            num_classes = int(graph.labels.shape[1])
        else:
            num_classes = int(graph.labels.max().item()+1)
            task_name = 'node'

        centroids = []
        pos_link = []
        neg_link = []
        for i in range(num_classes):
            lab = graph.labels[graph.train_labels]

            if task_name == 'node':
                tmp = np.where((lab == i).detach().cpu().numpy())[0].tolist()
                nodes = [graph.train_labels[_] for _ in tmp]

                for node in nodes:
                    for j in range(num_classes):
                        if j == i:pos_link.append((node,j+graph.x.size(0)))
                        else: neg_link.append((node,j+graph.x.size(0)))
                if len(tmp) == 0: centroids.append(graph.x[graph.train_labels].mean(0))
                else: centroids.append(graph.x[graph.train_labels][tmp].mean(0))
            if task_name == 'node_m':
                tmp = np.where((lab[:,i]==1).detach().cpu().numpy())[0].tolist()
                nodes = [graph.train_labels[_] for _ in tmp]
                for j in graph.train_labels:
                    if j in nodes:pos_link.append((j,i+graph.x.size(0)))
                    else: neg_link.append((j,i+graph.x.size(0)))
                if len(tmp) == 0:centroids.append(graph.x[graph.train_labels].mean(0))
                else: centroids.append(graph.x[graph.train_labels][lab[:,i]==1].mean(0))


        graph.x = torch.vstack([graph.x]+centroids)
        pos_edge_index = torch.tensor(np.array(pos_link).T).to(device)
        neg_edge_index = torch.tensor(np.array(neg_link).T).to(device)
        additional_edge_index = torch.tensor(np.array(pos_link+[(j,i) for i,j in pos_link]).T).to(device)


    result = []
    for epoch in range(1,2000):


        model.train()
        if mlp!= None:mlp.train()

        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
            else:z = model.encode(mlp(graph.x), graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss3(z, pos_edge_index,neg_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        if mlp!= None:mlp.eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())

            if mlp!= None:
                if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
                else:z = model.encode(mlp(graph.x), graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else:
                score = model.class_test(z,[_ for _ in range(graph.x.size(0)-num_classes,graph.x.size(0))],graph.test_labels ,graph.labels[graph.test_labels])


        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]:print(epoch,value,loss.item())
        result.append(score)


        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result


def train_graph_by_link(mlp,model, graph,optimizer,device,file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0

    if model.task != 'link':

        if len(graph.labels.shape) == 2:
            task_name = 'node_m'
            num_classes = int(graph.labels.shape[1])
        else:
            num_classes = int(graph.labels.max().item()+1)
            task_name = 'node'

        centroids = []
        pos_link = []
        neg_link = []
        for i in range(num_classes):
            lab = graph.labels[graph.train_labels]

            if task_name == 'node':
                tmp = np.where((lab == i).detach().cpu().numpy())[0].tolist()
                nodes = [graph.train_labels[_] for _ in tmp]

                for node in nodes:
                    for j in range(num_classes):
                        if j == i:pos_link.append((node,j+graph.x.size(0)))
                        else: neg_link.append((node,j+graph.x.size(0)))
                if len(tmp) == 0: centroids.append(graph.x[graph.train_labels].mean(0))
                else: centroids.append(graph.x[graph.train_labels][tmp].mean(0))
            if task_name == 'node_m':
                tmp = np.where((lab[:,i]==1).detach().cpu().numpy())[0].tolist()
                nodes = [graph.train_labels[_] for _ in tmp]
                for j in graph.train_labels:
                    if j in nodes:pos_link.append((j,i+graph.x.size(0)))
                    else: neg_link.append((j,i+graph.x.size(0)))
                if len(tmp) == 0:centroids.append(graph.x[graph.train_labels].mean(0))
                else: centroids.append(graph.x[graph.train_labels][lab[:,i]==1].mean(0))


        graph.x = torch.vstack([graph.x]+centroids)
        pos_edge_index = torch.tensor(np.array(pos_link).T).to(device)
        neg_edge_index = torch.tensor(np.array(neg_link).T).to(device)
        additional_edge_index = torch.tensor(np.array(pos_link+[(j,i) for i,j in pos_link]).T).to(device)
        graph.edge_index = torch.hstack([graph.edge_index,additional_edge_index])

    result = []
    for epoch in range(1,1000):


        model.train()
        if mlp!= None:mlp.train()

        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
            else:z = model.encode(mlp(graph.x), graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss(z, pos_edge_index,neg_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        if mlp!= None:mlp.eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())

            if mlp!= None:
                if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
                else:z = model.encode(mlp(graph.x), graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else:
                score = model.class_test(z,[_ for _ in range(graph.x.size(0)-num_classes,graph.x.size(0))],graph.test_labels ,graph.labels[graph.test_labels])


        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,value,loss.item())


        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result


def train_graph_by_link2(mlp,model, graph,optimizer, device,centroids =None,file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0

    num_classes = centroids.size(0)


    result = []
    for epoch in range(1,1000):


        model.train()
        if mlp!= None:mlp.train()

        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights, inner_loop=True)
            else:
                matrix = torch.vstack([mlp(graph.x),centroids])
                z = model.encode(matrix, graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:
            if epoch<=50:loss = model.class_loss(z, graph.pos_edge_index,graph.neg_edge_index)
            else:loss = model.class_loss2(z, graph.pos_edge_index,graph.neg_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)


        model.eval()

        if mlp!= None:mlp.eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())

            if mlp!= None:
                if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights, inner_loop=True)
                else:
                    matrix = torch.vstack([mlp(graph.x),centroids])
                    z = model.encode(matrix, graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else:
                score = model.class_test(z,[_ for _ in range(matrix.size(0)-num_classes,matrix.size(0))],graph.test_labels ,graph.labels[graph.test_labels])


        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,value,loss.item())


        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result


def lamp_gnn_gradient_step(args, pretrain_model,fake_model, wdiscriminator, syn_graphs, subgraphs, modality_projections,feature_MLPs,optimizer_wd, optimizer_all,optimizer_feats,epoch, device,file):

    task_losses = []

    torch.autograd.set_detect_anomaly(True)


    graphs = syn_graphs.transformation(feature_MLPs)

    Dis_loss_ = torch.tensor(0.0).to(device)
    for id,name in enumerate(args.train_dataset):

        data_graph = subgraphs[name]

        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(data_graph.transformation(modality_projections[name]), graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1s = []
        for graph in graphs:w1s.append(wdiscriminator_copy(graph.x.detach(),graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(modality_projections[name](data_graph.x), data_graph.edge_index)

        Dis_loss_ += (torch.mean(w1) - torch.mean(w0))


    optimizer_all.zero_grad()
    Dis_loss_.backward()
    optimizer_all.step()


    pretrain_model.train()
    fake_model.train()
    fake_model.reset_parameters()

    aucs = []
    aps = []
    fast_weights = OrderedDict(fake_model.named_parameters())
    with torch.no_grad():
        for id,name in enumerate(args.train_dataset):
            data_graph = subgraphs[name]
            z_val = fake_model.encode(modality_projections[name](data_graph.x), data_graph.edge_index , fast_weights)
            aucb,_ = fake_model.test(z_val,data_graph.edge_index, negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
            aps.append(str(round(aucb,3)))


    for inner_batch in range(5):

        loss = torch.tensor(0.0).to(device)
        for index in range(args.syn_graph_num):
            syn_graph= syn_graphs.syn_graphs[index]
            mlp = feature_MLPs[index]
            z = fake_model.encode(mlp(syn_graph.x), syn_graph.edge_index,fast_weights)
            loss += fake_model.recon_loss(z,syn_graph.edge_index)
        print(loss.item())

        gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
        gradients = [0 if grad is None else grad for grad in gradients]


        fast_weights = OrderedDict(
            (name, torch.clamp((param - 0.05 * grad), -0.1, 0.1))
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
    Syn_loss = torch.tensor(0.0).to(device)
    for id,name in enumerate(args.train_dataset):
        data_graph = subgraphs[name]
        z_val = fake_model.encode(modality_projections[name](data_graph.x).detach(), data_graph.edge_index , fast_weights)

        Syn_loss+=fake_model.recon_loss(z_val, data_graph.edge_index)

        with torch.no_grad():auc, ap = fake_model.test(z_val,data_graph.edge_index, negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
        aucs.append(str(round(auc,3)))


    Link_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph= syn_graphs.syn_graphs[index]
        mlp = feature_MLPs[index]
        syn_graph.link_split(0.2,0.2)
        weights = OrderedDict(pretrain_model.named_parameters())
        fast_weights =  pretrain_model.encoder.modulate(mlp(syn_graph.x).detach(), syn_graph.train_edge_index,weights)

        for inner_batch in range(args.inner_train_steps):

            z = pretrain_model.encode(mlp(syn_graph.x).detach(), syn_graph.train_edge_index,fast_weights)
            loss = pretrain_model.recon_loss(z,syn_graph.train_edge_index)

            gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
            gradients = [0 if grad is None else grad for grad in gradients]

            fast_weights = OrderedDict(
                    (name, torch.clamp((param - args.inner_lr * grad), -0.1, 0.1))
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        z_val = pretrain_model.encode(mlp(syn_graph.x), syn_graph.test_edge_index , fast_weights)
        Link_loss+=pretrain_model.recon_loss(z_val, syn_graph.test_edge_index)


    graphs = [subgraphs[name].transformation(modality_projections[name]) for name in args.train_dataset]
    Dis_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph= syn_graphs.syn_graphs[index]

        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(syn_graph.transformation(feature_MLPs[index]), graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1s = []
        for graph in graphs:w1s.append(wdiscriminator_copy(graph.x.detach(),graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(feature_MLPs[index](syn_graph.x), syn_graph.edge_index)

        Dis_loss += (torch.mean(w1) - torch.mean(w0))


    loss =  args.beta*Link_loss + args.alpha*Dis_loss + Syn_loss


    task_losses.append(loss)

    file.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n'.format(epoch, '-'.join(args.train_dataset),Syn_loss.item(),\
            Link_loss.item(),Dis_loss.item(),Dis_loss_.item(),'-'.join(aps),'-'.join(aucs)))
    file.flush()


    if len(task_losses) != 0:


        optimizer_feats.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()


        optimizer_feats.step()


    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)


    return pretrain_model,Link_loss.item()


def lamp_gnn_gradient_step_wo_appr(args, pretrain_model, fake_model, wdiscriminator, syn_graphs, subgraphs, modality_projections, feature_MLPs,
                           optimizer_wd, optimizer_all, optimizer_feats, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    graphs = syn_graphs.transformation(feature_MLPs)

    Dis_loss_ = torch.tensor(0.0).to(device)
    for id, name in enumerate(args.train_dataset):

        data_graph = subgraphs[name]

        wdiscriminator_copy = copy.deepcopy(
            train_wdiscriminator(data_graph.transformation(modality_projections[name]), graphs, wdiscriminator,
                                 optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)

        w1s = []
        for graph in graphs: w1s.append(wdiscriminator_copy(graph.x.detach(), graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(modality_projections[name](data_graph.x), data_graph.edge_index)

        Dis_loss_ += (torch.mean(w1) - torch.mean(w0))

    optimizer_all.zero_grad()
    Dis_loss_.backward()
    optimizer_all.step()

    pretrain_model.train()
    fake_model.train()
    fake_model.reset_parameters()


    Link_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph = syn_graphs.syn_graphs[index]
        mlp = feature_MLPs[index]
        syn_graph.link_split(0.2, 0.2)
        weights = OrderedDict(pretrain_model.named_parameters())
        fast_weights = pretrain_model.encoder.modulate(mlp(syn_graph.x).detach(), syn_graph.train_edge_index, weights)

        for inner_batch in range(args.inner_train_steps):
            z = pretrain_model.encode(mlp(syn_graph.x).detach(), syn_graph.train_edge_index, fast_weights)
            loss = pretrain_model.recon_loss(z, syn_graph.train_edge_index)

            gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
            gradients = [0 if grad is None else grad for grad in gradients]

            fast_weights = OrderedDict(
                (name, torch.clamp((param - args.inner_lr * grad), -0.1, 0.1))
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        z_val = pretrain_model.encode(mlp(syn_graph.x), syn_graph.test_edge_index, fast_weights)
        Link_loss += pretrain_model.recon_loss(z_val, syn_graph.test_edge_index)

    graphs = [subgraphs[name].transformation(modality_projections[name]) for name in args.train_dataset]
    Dis_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph = syn_graphs.syn_graphs[index]

        wdiscriminator_copy = copy.deepcopy(
            train_wdiscriminator(syn_graph.transformation(feature_MLPs[index]), graphs, wdiscriminator,
                                 optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)

        w1s = []
        for graph in graphs: w1s.append(wdiscriminator_copy(graph.x.detach(), graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(feature_MLPs[index](syn_graph.x), syn_graph.edge_index)

        Dis_loss += (torch.mean(w1) - torch.mean(w0))

    loss = args.beta * Link_loss + args.alpha * Dis_loss

    task_losses.append(loss)

    file.write('{},{},{:.3f},{:.3f},{:.3f}\n'.format(epoch, '-'.join(args.train_dataset), \
                                                                  Link_loss.item(), Dis_loss.item(), Dis_loss_.item()))
    file.flush()

    if len(task_losses) != 0:
        optimizer_feats.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer_feats.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, Link_loss.item()


def lamp_gnn_gradient_step_wo_meta(args, pretrain_model, fake_model, wdiscriminator, syn_graphs, subgraphs, modality_projections, feature_MLPs,
                           optimizer_wd, optimizer_all, optimizer_feats, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    graphs = syn_graphs.transformation(feature_MLPs)

    Dis_loss_ = torch.tensor(0.0).to(device)
    for id, name in enumerate(args.train_dataset):

        data_graph = subgraphs[name]

        wdiscriminator_copy = copy.deepcopy(
            train_wdiscriminator(data_graph.transformation(modality_projections[name]), graphs, wdiscriminator,
                                 optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)

        w1s = []
        for graph in graphs: w1s.append(wdiscriminator_copy(graph.x.detach(), graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(modality_projections[name](data_graph.x), data_graph.edge_index)

        Dis_loss_ += (torch.mean(w1) - torch.mean(w0))

    optimizer_all.zero_grad()
    Dis_loss_.backward()
    optimizer_all.step()

    pretrain_model.train()
    fake_model.train()
    fake_model.reset_parameters()

    aucs = []
    aps = []
    fast_weights = OrderedDict(fake_model.named_parameters())
    with torch.no_grad():
        for id, name in enumerate(args.train_dataset):
            data_graph = subgraphs[name]
            z_val = fake_model.encode(modality_projections[name](data_graph.x), data_graph.edge_index, fast_weights)
            aucb, _ = fake_model.test(z_val, data_graph.edge_index,
                                      negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
            aps.append(str(round(aucb, 3)))

    for inner_batch in range(5):

        loss = torch.tensor(0.0).to(device)
        for index in range(args.syn_graph_num):
            syn_graph = syn_graphs.syn_graphs[index]
            mlp = feature_MLPs[index]
            z = fake_model.encode(mlp(syn_graph.x), syn_graph.edge_index, fast_weights)
            loss += fake_model.recon_loss(z, syn_graph.edge_index)
        print(loss.item())

        gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                        allow_unused=True, create_graph=True)
        gradients = [0 if grad is None else grad for grad in gradients]


        fast_weights = OrderedDict(
            (name, torch.clamp((param - 0.05 * grad), -0.1, 0.1))
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
    Syn_loss = torch.tensor(0.0).to(device)

    for id, name in enumerate(args.train_dataset):
        data_graph = subgraphs[name]
        z_val = fake_model.encode(modality_projections[name](data_graph.x).detach(), data_graph.edge_index, fast_weights)

        Syn_loss += fake_model.recon_loss(z_val, data_graph.edge_index)

        with torch.no_grad(): auc, ap = fake_model.test(z_val, data_graph.edge_index,
                                                        negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
        aucs.append(str(round(auc, 3)))

    Link_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph = syn_graphs.syn_graphs[index]
        mlp = feature_MLPs[index]
        syn_graph.link_split(0.2, 0.2)
        weights = OrderedDict(pretrain_model.named_parameters())
        fast_weights = pretrain_model.encoder.modulate(mlp(syn_graph.x).detach(), syn_graph.train_edge_index, weights)
        z_val = pretrain_model.encode(mlp(syn_graph.x), syn_graph.train_edge_index, fast_weights)
        Link_loss += pretrain_model.recon_loss(z_val, syn_graph.test_edge_index)

    graphs = [subgraphs[name].transformation(modality_projections[name]) for name in args.train_dataset]
    Dis_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph = syn_graphs.syn_graphs[index]

        wdiscriminator_copy = copy.deepcopy(
            train_wdiscriminator(syn_graph.transformation(feature_MLPs[index]), graphs, wdiscriminator,
                                 optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)

        w1s = []
        for graph in graphs: w1s.append(wdiscriminator_copy(graph.x.detach(), graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(feature_MLPs[index](syn_graph.x), syn_graph.edge_index)

        Dis_loss += (torch.mean(w1) - torch.mean(w0))

    loss = args.beta * Link_loss + args.alpha * Dis_loss + Syn_loss

    task_losses.append(loss)

    file.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n'.format(epoch, '-'.join(args.train_dataset), Syn_loss.item(), \
                                                                  Link_loss.item(), Dis_loss.item(), Dis_loss_.item(),
                                                                  '-'.join(aps), '-'.join(aucs)))
    file.flush()

    if len(task_losses) != 0:
        optimizer_feats.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer_feats.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, Link_loss.item()


def lamp_gnn_gradient_step_wo_dist(args, pretrain_model, fake_model, wdiscriminator, syn_graphs, subgraphs, modality_projections, feature_MLPs,
                           optimizer_wd, optimizer_all, optimizer_feats, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    graphs = syn_graphs.transformation(feature_MLPs)


    aucs = []
    aps = []
    fast_weights = OrderedDict(fake_model.named_parameters())
    with torch.no_grad():
        for id, name in enumerate(args.train_dataset):
            data_graph = subgraphs[name]
            z_val = fake_model.encode(modality_projections[name](data_graph.x), data_graph.edge_index, fast_weights)
            aucb, _ = fake_model.test(z_val, data_graph.edge_index,
                                      negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
            aps.append(str(round(aucb, 3)))

    for inner_batch in range(5):

        loss = torch.tensor(0.0).to(device)
        for index in range(args.syn_graph_num):
            syn_graph = syn_graphs.syn_graphs[index]
            mlp = feature_MLPs[index]
            z = fake_model.encode(mlp(syn_graph.x), syn_graph.edge_index, fast_weights)
            loss += fake_model.recon_loss(z, syn_graph.edge_index)
        print(loss.item())

        gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                        allow_unused=True, create_graph=True)
        gradients = [0 if grad is None else grad for grad in gradients]


        fast_weights = OrderedDict(
            (name, torch.clamp((param - 0.05 * grad), -0.1, 0.1))
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
    Syn_loss = torch.tensor(0.0).to(device)
    for id, name in enumerate(args.train_dataset):
        data_graph = subgraphs[name]
        z_val = fake_model.encode(modality_projections[name](data_graph.x).detach(), data_graph.edge_index, fast_weights)

        Syn_loss += fake_model.recon_loss(z_val, data_graph.edge_index)

        with torch.no_grad(): auc, ap = fake_model.test(z_val, data_graph.edge_index,
                                                        negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
        aucs.append(str(round(auc, 3)))

    Link_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph = syn_graphs.syn_graphs[index]
        mlp = feature_MLPs[index]
        syn_graph.link_split(0.2, 0.2)
        weights = OrderedDict(pretrain_model.named_parameters())
        fast_weights = pretrain_model.encoder.modulate(mlp(syn_graph.x).detach(), syn_graph.train_edge_index, weights)

        for inner_batch in range(args.inner_train_steps):
            z = pretrain_model.encode(mlp(syn_graph.x).detach(), syn_graph.train_edge_index, fast_weights)
            loss = pretrain_model.recon_loss(z, syn_graph.train_edge_index)

            gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
            gradients = [0 if grad is None else grad for grad in gradients]

            fast_weights = OrderedDict(
                (name, torch.clamp((param - args.inner_lr * grad), -0.1, 0.1))
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        z_val = pretrain_model.encode(mlp(syn_graph.x), syn_graph.test_edge_index, fast_weights)
        Link_loss += pretrain_model.recon_loss(z_val, syn_graph.test_edge_index)


    loss = args.beta * Link_loss  + Syn_loss

    task_losses.append(loss)

    file.write('{},{},{:.3f},{:.3f},{},{}\n'.format(epoch, '-'.join(args.train_dataset), Syn_loss.item(), \
                                                                  Link_loss.item(),
                                                                  '-'.join(aps), '-'.join(aucs)))
    file.flush()

    if len(task_losses) != 0:
        optimizer_feats.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer_feats.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, Link_loss.item()


def lamp_gnn_gradient_step_test_wo(args, pretrain_model,fake_model, wdiscriminator, syn_graphs, subgraphs, modality_projections,feature_MLPs,optimizer_wd, optimizer_all,optimizer_feats,epoch, device,file):

    task_losses = []

    torch.autograd.set_detect_anomaly(True)


    graphs = syn_graphs.transformation(feature_MLPs)

    Dis_loss_ = torch.tensor(0.0).to(device)
    for id,name in enumerate(args.train_dataset):

        data_graph = subgraphs[name]

        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(data_graph.transformation(modality_projections[name]), graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1s = []
        for graph in graphs:w1s.append(wdiscriminator_copy(graph.x.detach(),graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(modality_projections[name](data_graph.x), data_graph.edge_index)

        Dis_loss_ += (torch.mean(w1) - torch.mean(w0))


    optimizer_all.zero_grad()
    Dis_loss_.backward()
    optimizer_all.step()


    pretrain_model.train()
    fake_model.train()
    fake_model.reset_parameters()

    aucs = []
    aps = []
    fast_weights = OrderedDict(fake_model.named_parameters())
    with torch.no_grad():
        for id,name in enumerate(args.train_dataset):
            data_graph = subgraphs[name]
            z_val = fake_model.encode(modality_projections[name](data_graph.x), data_graph.edge_index , fast_weights)
            aucb,_ = fake_model.test(z_val,data_graph.edge_index, negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
            aps.append(str(round(aucb,3)))


    for inner_batch in range(5):

        loss = torch.tensor(0.0).to(device)
        for index in range(args.syn_graph_num):
            syn_graph= syn_graphs.syn_graphs[index]
            mlp = feature_MLPs[index]
            z = fake_model.encode(mlp(syn_graph.x), syn_graph.edge_index,fast_weights)
            loss += fake_model.recon_loss(z,syn_graph.edge_index)
        print(loss.item())

        gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
        gradients = [0 if grad is None else grad for grad in gradients]


        fast_weights = OrderedDict(
            (name, torch.clamp((param - 0.05 * grad), -0.1, 0.1))
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
    Syn_loss = torch.tensor(0.0).to(device)
    for id,name in enumerate(args.train_dataset):
        data_graph = subgraphs[name]
        z_val = fake_model.encode(modality_projections[name](data_graph.x).detach(), data_graph.edge_index , fast_weights)

        Syn_loss+=fake_model.recon_loss(z_val, data_graph.edge_index)

        with torch.no_grad():auc, ap = fake_model.test(z_val,data_graph.edge_index, negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
        aucs.append(str(round(auc,3)))


    Link_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph= syn_graphs.syn_graphs[index]
        mlp = feature_MLPs[index]
        syn_graph.link_split(0.2,0.2)
        weights = OrderedDict(pretrain_model.named_parameters())
        fast_weights =  weights

        for inner_batch in range(args.inner_train_steps):

            z = pretrain_model.encode(mlp(syn_graph.x).detach(), syn_graph.train_edge_index,fast_weights)
            loss = pretrain_model.recon_loss(z,syn_graph.train_edge_index)

            gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
            gradients = [0 if grad is None else grad for grad in gradients]

            fast_weights = OrderedDict(
                    (name, torch.clamp((param - args.inner_lr * grad), -0.1, 0.1))
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        z_val = pretrain_model.encode(mlp(syn_graph.x), syn_graph.test_edge_index , fast_weights)
        Link_loss+=pretrain_model.recon_loss(z_val, syn_graph.test_edge_index)


    graphs = [subgraphs[name].transformation(modality_projections[name]) for name in args.train_dataset]
    Dis_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph= syn_graphs.syn_graphs[index]

        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(syn_graph.transformation(feature_MLPs[index]), graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1s = []
        for graph in graphs:w1s.append(wdiscriminator_copy(graph.x.detach(),graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(feature_MLPs[index](syn_graph.x), syn_graph.edge_index)

        Dis_loss += (torch.mean(w1) - torch.mean(w0))


    loss =  Link_loss + Dis_loss + Syn_loss


    task_losses.append(loss)

    file.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n'.format(epoch, '-'.join(args.train_dataset),Syn_loss.item(),\
            Link_loss.item(),Dis_loss.item(),Dis_loss_.item(),'-'.join(aps),'-'.join(aucs)))
    file.flush()


    if len(task_losses) != 0:


        optimizer_feats.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()


        optimizer_feats.step()


    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)


    return pretrain_model,Link_loss.item()


def MDP_gradient_step_original(args, task, pretrain_model, auxiliary, graphs, optimizer, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()
    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model, auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x, graph.edge_index, fast_weights).mean(0)
        return auxiliary(x).detach()

    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model, auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape, x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters(): p.data.clamp_(-0.1, 0.1)
        return pretrain_model, loss.item()


    for index,name in enumerate( graphs.keys()):

        graph = graphs[name]


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask], graph.attr)

        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        task_losses.append(loss)

    if len(task_losses) != 0:
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def findsamples(adj):
    nodenum=adj.size(0)
    result=np.ones((nodenum,3))
    src=adj._indices()[1].tolist()
    dst=adj._indices()[0].tolist()


    for i in range(nodenum):
        result[i,0]=i
        if i not in src:
            result[i,1]=i
        else:
            index_i=src.index(i)
            i_point_to=dst[index_i]
            result[i,1]=i_point_to

        for node in range(nodenum):
            if not adj[i,node] and  i != node:
                result[i,2]=node
                break
    return torch.tensor(result,dtype=int)


def pretrain_GP_step(args, pretrain_model, graphs,optimizer, epoch,device):

    task_losses = []
    torch.autograd.set_detect_anomaly(True)
    pretrain_model.train()
    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())
    for graph in graphs.values():
        adj = torch.sparse_coo_tensor(indices = np.array([graph.adj.row,graph.adj.col]),values = np.ones_like(graph.adj.row),size = (graph.x.shape[0],graph.x.shape[0]))
        samples = findsamples(adj)
        pred = pretrain_model.encode(graph.x, graph.edge_index , weights)
        pred=torch.sigmoid(pred).double()
        size_ = adj.size(0)
        p = [i for i in range(size_)]
        x = [p,p]
        q = [1 for i in range(size_)]

        tt = torch.sparse_coo_tensor(x,q,(size_,size_))
        adj = (adj + tt).double().to(device)
        pred = torch.matmul(adj, pred)
        node_feature = pred[samples]
        a = node_feature[:,0,:]
        positive = node_feature[:,1,:]
        negative =  node_feature[:,2,:]
        x = torch.exp(F.cosine_similarity(a,positive,dim=-1)/1)
        y = torch.exp(F.cosine_similarity(a,negative,dim=-1)/1)

        loss=(-1*torch.log(x/y)).mean()
        task_losses.append(loss)


    if len(task_losses) != 0:

        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model,pretrain_batch_loss.item()

def MDP_gradient_step(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections, epoch, device, file,flag = False):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()
    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())


    for index,name in enumerate( graphs.keys()):


        if task == 'EdgePred':
            graph = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])

        elif task == 'sim':


            graph = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))

            vice_model = gen_ran_output(pretrain_model)
            output_negative = vice_model.encoder(graph.x, graph.edge_index, weights)
            output_positive= pretrain_model.encoder(graph.x, graph.edge_index, weights)

            loss = loss_cal2(output_positive,output_negative,auxiliary)


        else:
            graph = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        task_losses.append(loss)

    if len(task_losses) != 0:
        optimizer.zero_grad()
        stack = torch.stack(task_losses)
        if flag: pretrain_batch_loss = stack.mean()
        else: pretrain_batch_loss = stack.mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_step2(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model, auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x, graph.edge_index, fast_weights).mean(0)
        return auxiliary(x).detach()

    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model, auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape, x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters(): p.data.clamp_(-0.1, 0.1)
        return pretrain_model, loss.item()


    for index,name in enumerate( graphs.keys()):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    task_meta_losses = []
    for _ in range(1):
        ratios = [np.random.rand() for _ in range(len(args.train_dataset))]
        ratios = np.array([0.25]*4)
        print(ratios)
        adj = 0.0
        x = 0.0
        for index,name in enumerate(args.train_dataset):
            adj+=ratios[index]*graphs[name].adj.toarray().astype(np.bool)
            x+=ratios[index]*graphs[name].x


        meta_graph_adj= adj
        meta_graph_x= x.detach()/(len(graphs)+0.0)

        print(len(meta_graph_adj.nonzero()[1]))

        meta_graph_adj[meta_graph_adj<0.5] = 0
        meta_graph_adj[meta_graph_adj>=0.5] = 1


        meta_graph_adj = coo_array(meta_graph_adj)

        meta_graph_adj_ = np.vstack((meta_graph_adj.row, meta_graph_adj.col)).astype(np.long)
        print(meta_graph_adj_.shape)
        meta_graph = Graph(meta_graph_x, torch.from_numpy(meta_graph_adj_).to(device))


        if task == 'EdgePred':
            meta_graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(meta_graph.x, meta_graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, meta_graph.test_edge_index)

        elif task == 'AttrMask':
            meta_graph.attr_split(0.2)
            z_val = pretrain_model.encode(meta_graph.x, meta_graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[meta_graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], meta_graph.original_x[meta_graph.mask])


        else:
            output_positive = pretrain_model.encode(meta_graph.x, meta_graph.edge_index, weights)
            arr = torch.arange(meta_graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(meta_graph.x[arr], meta_graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

        print(loss.item())
        task_meta_losses.append(loss)


    task_losses = []
    for index,name in enumerate( graphs.keys()):

        graph = graphs[name]


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])


        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        print(loss.item())
        task_losses.append(loss)

    if len(task_losses) != 0:
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()+torch.stack(task_meta_losses).mean()

        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def pretrain_gcope_step(args, pretrain_model,auxiliary1,auxiliary2 ,graphs,optimizer, epoch,device,file):


    torch.autograd.set_detect_anomaly(True)


    pretrain_model.train()
    auxiliary1.train()
    auxiliary2.train()
    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())

    def gen_ran_output( model):
        vice_model = deepcopy(model)

        for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):

            adv_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(param.data) * param.data.std())
        return vice_model

    vice_model = gen_ran_output(pretrain_model)
    x2s = []
    x1s = []
    xs = []
    for graph in graphs:
        xs.append(graph.x)
        x2s.append(auxiliary1(vice_model.encoder(graph.x,graph.edge_index,weights)))
        x1s.append(auxiliary1(pretrain_model.encoder(graph.x,graph.edge_index,weights)))


    x2s = torch.stack(x2s)
    x1s = torch.stack(x1s)

    x2 =  torch.stack([_.mean(0) for _ in x2s]).detach()
    x1 =  torch.stack([_.mean(0) for _ in x1s])
    xs = torch.stack(xs)


    print(x1.shape,x2.shape)
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / 0.1)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (1e-4+sim_matrix.sum(dim=1))
    loss_cl = - torch.log(loss).mean()


    loss_cons = torch.nn.MSELoss()(auxiliary2(x2s),xs)+torch.nn.MSELoss()(auxiliary2(x1s),xs)
    loss = loss_cl+0.2*loss_cons


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for p in pretrain_model.parameters():p.data.clamp_(-0.1, 0.1)
    return pretrain_model,loss.item()


def pretrain_task_step(args,task, pretrain_model,auxiliary, graphs,optimizer, epoch,device,file):


    task_losses = []

    torch.autograd.set_detect_anomaly(True)


    pretrain_model.train()
    if auxiliary: auxiliary.train()
    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model,auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x,graph.edge_index,fast_weights).mean(0)
        return auxiliary(x).detach()


    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model,auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x,graph.edge_index,weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape,x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (1e-4+sim_matrix.sum(dim=1))
        loss = - torch.log(loss).mean()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters():p.data.clamp_(-0.1, 0.1)
        return pretrain_model,loss.item()


    for graph in graphs.values():
        if task == 'EdgePred':
            graph.link_split(0.2,0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index , weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index , weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask],graph.attr)

        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index , weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index , weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive,dim=0,keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive@discriminator_summary
            negative_score = output_negative@discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score, torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(negative_score, torch.zeros_like(negative_score))
        task_losses.append(loss)


    file.write('{},{},{:.3f},{:.3f},{:.3f}\n'.format(epoch, '-'.join(args.train_dataset),\
            task_losses[0].item(),task_losses[1].item(),task_losses[2].item()))
    file.flush()

    if len(task_losses) != 0:

        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)


    return pretrain_model,pretrain_batch_loss.item()


def MDP_gradient_step_mix_partial(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections,wdiscriminator,optimizer_wd, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    weights = OrderedDict(pretrain_model.named_parameters())


    task_meta_losses = []
    for _ in range(1):

        num = args.subgraph_scale-int(args.subgraph_scale*0.2)

        mix_adj = np.zeros([int(args.subgraph_scale*0.2)+4*num,int(args.subgraph_scale*0.2)+4*num])


        ratios = np.array([0.25]*4)
        print(ratios)
        adj = graphs[args.train_dataset[0]].adj.toarray()
        x = graphs[args.train_dataset[0]].x

        mix_adj[:args.subgraph_scale,:][:,:args.subgraph_scale] = adj
        mix_x = x

        adj = ratios[0]*adj[:int(args.subgraph_scale*0.2),:][:,:int(args.subgraph_scale*0.2)]
        x = ratios[0]*x[:int(args.subgraph_scale*0.2),:]


        for index in range(1,len(args.train_dataset)):

            tmp = train_wdiscriminator_align(graphs[args.train_dataset[index]], graphs[args.train_dataset[0]], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80)


            adj_=graphs[args.train_dataset[index]].adj.toarray()[tmp,:][:,tmp]
            x_=graphs[args.train_dataset[index]].x[tmp,:]

            adj+=ratios[0]*adj_[:int(args.subgraph_scale*0.2),:][:,:int(args.subgraph_scale*0.2)]
            x+=ratios[0]*x_[:int(args.subgraph_scale*0.2),:]

            mix_adj[:args.subgraph_scale,:][int(args.subgraph_scale*0.2)+index*num:int(args.subgraph_scale*0.2)+(1+index)*num]\
            =adj_[:args.subgraph_scale,:][:,args.subgraph_scale:]

            mix_adj[int(args.subgraph_scale*0.2)+index*num:int(args.subgraph_scale*0.2)+(1+index)*num][:args.subgraph_scale,:]\
            =adj_[:,args.subgraph_scale:][:args.subgraph_scale,:]

            mix_adj[int(args.subgraph_scale*0.2)+index*num:int(args.subgraph_scale*0.2)+(1+index)*num]\
                [int(args.subgraph_scale*0.2)+index*num:int(args.subgraph_scale*0.2)+(1+index)*num] \
                =adj_[:,args.subgraph_scale:][args.subgraph_scale:,:]


            mix_x = torch.cat((mix_x,x_[args.subgraph_scale:]), dim = 0)


        adj[adj>=0.5] = 1
        adj[adj<0.5] = 0

        mix_adj[:args.subgraph_scale,:][:,:args.subgraph_scale] = adj
        mix_x[:args.subgraph_scale] = x


        meta_graph_adj= mix_adj
        meta_graph_x= mix_x.detach()


        meta_graph_adj = coo_array(meta_graph_adj)

        meta_graph_adj_ = np.vstack((meta_graph_adj.row, meta_graph_adj.col)).astype(np.long)
        print(meta_graph_adj_.shape)
        meta_graph = Graph(meta_graph_x, torch.from_numpy(meta_graph_adj_).to(device))


        if task == 'EdgePred':
            meta_graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(meta_graph.x, meta_graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, meta_graph.test_edge_index)

        elif task == 'AttrMask':
            meta_graph.attr_split(0.2)
            z_val = pretrain_model.encode(meta_graph.x, meta_graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[meta_graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], meta_graph.original_x[meta_graph.mask])


        else:
            output_positive = pretrain_model.encode(meta_graph.x, meta_graph.edge_index, weights)
            arr = torch.arange(meta_graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(meta_graph.x[arr], meta_graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

        print(loss.item())
        task_meta_losses.append(loss)


def MDP_gradient_step_mix(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections,wdiscriminator,optimizer_wd, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    weights = OrderedDict(pretrain_model.named_parameters())


    task_meta_losses = []
    for _ in range(1):


        ratios = np.array([0.25]*4)
        print(ratios)
        adj = ratios[0]*graphs[args.train_dataset[0]].adj.toarray()
        x = ratios[0]*graphs[args.train_dataset[0]].x


        for index in range(1,len(args.train_dataset)):

            tmp = train_wdiscriminator_align(graphs[args.train_dataset[index]], graphs[args.train_dataset[0]], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80)


            adj+=ratios[index]*graphs[args.train_dataset[index]].adj.toarray()[tmp,:][:,tmp]

            x+=ratios[index]*graphs[args.train_dataset[index]].x[tmp,:]


        meta_graph_adj= adj
        meta_graph_x= x.detach()

        print(len(meta_graph_adj.nonzero()[1]))
        print(np.max(meta_graph_adj))


        meta_graph_adj[meta_graph_adj>=0.5] = 1
        meta_graph_adj[meta_graph_adj<0.5] = 0


        meta_graph_adj = coo_array(meta_graph_adj)

        meta_graph_adj_ = np.vstack((meta_graph_adj.row, meta_graph_adj.col)).astype(np.long)
        print(meta_graph_adj_.shape)
        meta_graph = Graph(meta_graph_x, torch.from_numpy(meta_graph_adj_).to(device))


        if task == 'EdgePred':
            meta_graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(meta_graph.x, meta_graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, meta_graph.test_edge_index)

        elif task == 'AttrMask':
            meta_graph.attr_split(0.2)
            z_val = pretrain_model.encode(meta_graph.x, meta_graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[meta_graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], meta_graph.original_x[meta_graph.mask])


        else:
            output_positive = pretrain_model.encode(meta_graph.x, meta_graph.edge_index, weights)
            arr = torch.arange(meta_graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(meta_graph.x[arr], meta_graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

        print(loss.item())
        task_meta_losses.append(loss)


    task_losses = []
    for index,name in enumerate( graphs.keys()):

        graph = graphs[name]


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])


        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        print(loss.item())
        task_losses.append(loss)

    if len(task_losses) != 0:
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()+torch.stack(task_meta_losses).mean()

        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_prompt_step(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()
    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model, auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x, graph.edge_index, fast_weights).mean(0)
        return auxiliary(x).detach()

    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model, auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape, x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters(): p.data.clamp_(-0.1, 0.1)
        return pretrain_model, loss.item()


    for index,name in enumerate( graphs.keys()):

        graph = graphs[name].promptamation(modality_projections,args.train_dataset,name)


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask], graph.attr)

        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        task_losses.append(loss)

    if len(task_losses) != 0:
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_step_wd(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections,wdiscriminator,optimizer_wd, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model, auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x, graph.edge_index, fast_weights).mean(0)
        return auxiliary(x).detach()

    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model, auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape, x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters(): p.data.clamp_(-0.1, 0.1)
        return pretrain_model, loss.item()


    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):
        graph= graphs[name]
        rest = list(graphs.keys())
        rest.remove(name)
        rest_graphs = [graphs[k] for k in rest]


        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, rest_graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1s = []
        for g in rest_graphs:w1s.append(wdiscriminator_copy(g.x,g.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(graph.x, graph.edge_index)

        Dis_loss = (torch.mean(w1) - torch.mean(w0))


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])

        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in task_losses])))
    file.flush()


    if len(task_losses) != 0:
        optimizer.zero_grad()
        stack = torch.stack(task_losses)
        pretrain_batch_loss = stack.mean()+torch.stack(dis_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_step_wd2(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections,wdiscriminator,optimizer_wd, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()

    adj = 0
    for index,name in enumerate(args.train_dataset):
        adj+=graphs[name].adj.toarray()
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    meta_graph_x = torch.stack([graphs[key].x for key in graphs.keys() ]).mean(0).detach()


    meta_graph_adj= adj/(len(graphs)+0.0)

    meta_graph_adj[meta_graph_adj<0.5] = 0
    meta_graph_adj[meta_graph_adj>=0.5] = 1


    meta_graph_adj = coo_array(meta_graph_adj)
    print(len(meta_graph_adj.data))

    meta_graph_adj_ = np.vstack((meta_graph_adj.row, meta_graph_adj.col)).astype(np.long)

    meta_graph = Graph(meta_graph_x, torch.from_numpy(meta_graph_adj_).to(device))


    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model, auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x, graph.edge_index, fast_weights).mean(0)
        return auxiliary(x).detach()

    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model, auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape, x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters(): p.data.clamp_(-0.1, 0.1)
        return pretrain_model, loss.item()


    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):
        graph= graphs[name]


        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, meta_graph, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=40))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)

        w1 = wdiscriminator_copy(meta_graph.x,meta_graph.edge_index)
        w0 = wdiscriminator_copy(graph.x, graph.edge_index)

        Dis_loss = (torch.mean(w1) - torch.mean(w0))


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])

        else:
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in task_losses])))
    file.flush()


    if len(task_losses) != 0:
        optimizer.zero_grad()

        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_step_wd3(args, task, pretrain_model, auxiliary, graphs, optimizer,modality_projections,wdiscriminator,optimizer_wd, epoch, device, file):
    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    task_losses = []

    weights = OrderedDict(pretrain_model.named_parameters())


    def gen_ran_output(graph, model, auxiliary):
        vice_model = deepcopy(model)

        fast_weights = OrderedDict(vice_model.named_parameters())
        for (name, param) in model.named_parameters():
            fast_weights[name].data.copy_(param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()))

        x = pretrain_model.encoder(graph.x, graph.edge_index, fast_weights).mean(0)
        return auxiliary(x).detach()

    if task == 'sim':
        x2s = []
        x1s = []
        for graph in graphs.values():
            x2s.append(gen_ran_output(graph, pretrain_model, auxiliary))
            x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
        x2 = torch.stack(x2s)
        x1 = torch.stack(x1s)


        print(x1.shape, x2.shape)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in pretrain_model.parameters(): p.data.clamp_(-0.1, 0.1)
        return pretrain_model, loss.item()


    for index,name in enumerate( graphs.keys()):
        graph= graphs[name]


        fake_model = copy.deepcopy(pretrain_model)
        for p in fake_model.parameters(): p.requires_grad = False

        source = fake_model.encode(graph.x, graph.edge_index, weights)
        target = torch.cat([fake_model.encode(graphs[g].x, graphs[g].edge_index, weights) for g in graphs],dim=0)


        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator_old(source.detach(), target.detach(), wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1 = wdiscriminator_copy(target)
        w0 = wdiscriminator_copy(source)

        Dis_loss = (torch.mean(w1) - torch.mean(w0))


        if task == 'EdgePred':
            graph.link_split(0.2, 0.2)
            z_val = pretrain_model.encode(graph.x.detach(), graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':
            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x.detach(), graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask], graph.attr.detach())

        else:
            output_positive = pretrain_model.encode(graph.x.detach(), graph.edge_index, weights)
            arr = torch.arange(graph.x.shape[0]) + 1
            arr[-1:] = torch.arange(1)
            output_negative = pretrain_model.encode(graph.x[arr].detach(), graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary
            loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))
        task_losses.append(Dis_loss*0.5+loss)
        print(Dis_loss.item(),loss.item())


    if len(task_losses) != 0:
        optimizer.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def fused_split(fused_graph,graphs,task):

    fused_graph.x = torch.cat([graphs[name].x for name in graphs.keys()],dim =0).detach()

    if task == 'EdgePred':
        cum = 0
        test_index = []

        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)


        for key in graphs.keys():
            graph = graphs[key]

            graph.link_split(0.2, 0.2)
            test_index.append(cum+graph.test_edge_index)

            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.train_edge_index[0]+cum,graph.train_edge_index[1]+cum] = 1.0

            cum+=graph.x.shape[0]


        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)

        fused_graph.adj_tensor = adj_tensor

        fused_graph.test_index = test_index


    elif task == 'AttrMask':

        cum = 0
        test_index = []
        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)
        for key in graphs.keys():
            graph = graphs[key]
            graph.attr_split2(0.2)
            test_index.append(graph.mask+cum)
            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]


        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)


        fused_graph.adj_tensor = adj_tensor
        fused_graph.test_index = test_index


        fused_graph.xx = torch.cat([graphs[name].xx for name in graphs.keys()],dim =0)


    elif task == 'sim':


        cum = 0

        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)
        for key in graphs.keys():
            graph = graphs[key]
            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)

        fused_graph.adj_tensor = adj_tensor


    else:

        all_edge_index = []

        cum = 0

        mask = torch.tensor(np.ones((fused_graph.adj.shape[0],fused_graph.adj.shape[1]))*2).to(fused_graph.adj)
        for key in graphs.keys():
            graph = graphs[key]

            all_edge_index.append(graph.edge_index+cum)

            mask[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            mask[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        adj_tensor = torch.where(mask==2,fused_graph.adj,mask)

        fused_graph.adj_tensor = adj_tensor
        fused_graph.original_adj = torch.cat(all_edge_index,dim=1)

        tmp = []
        node_nums = np.cumsum([0]+[graphs[name].x.shape[0] for name in graphs.keys()])
        for i in range(len(node_nums)-1):
            tmp.append(np.arange(node_nums[i],node_nums[i+1])+1)
            tmp[-1][-1] = node_nums[i]
        arr = np.concatenate(tmp)

        fused_graph.arr = arr


    return fused_graph,graphs


def lambda_training(args, task, base_model, base_auxiliary, fused_graph,graphs,device):

    models = []

    cum = 0
    for index,key in enumerate(graphs.keys()):
        graph = graphs[key]
        model = deepcopy(base_model),deepcopy(base_auxiliary)
        model_params = []
        model_params.extend(model[0].parameters())
        if model[1] is not None: model_params.extend(model[1].parameters())


        optimizer= torch.optim.Adam([{'params': model_params}], lr=0.0005 ,weight_decay=5e-4)   

        for epoch in range(10):
            model[0].train()
            if model[1] is not None:model[1].train()

            weights = OrderedDict(model[0].named_parameters())
            if task == 'EdgePred':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)

            elif task == 'sim':


                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())


                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())


                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])


            elif task == 'AttrMask':

                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1


                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


            loss = loss1+loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss1,loss2)
        cum+=graph.x.shape[0]


        for p in model[0].parameters(): p.requires_grad = False
        if model[1] is not None:
            for p in model[1].parameters(): p.requires_grad = False
        models.append(model)


    return models


def lambda_for_adj(args, task, base_model, base_auxiliary,graphs,match_graph,device):


    optimizer= torch.optim.Adam(match_graph.parameters(), lr=0.0001,weight_decay=5e-4)


    for loop in range(5):


        adj = match_graph(graphs)

        fused_graph = Graph(None,None)
        fused_graph.adj = adj


        fused_graph,graphs = fused_split(fused_graph,graphs,task)

        print(fused_graph.adj_tensor.sum(),fused_graph.adj_tensor)

        models = lambda_training(args, task, base_model, base_auxiliary, fused_graph,graphs,device)


        cum = 0
        loss = 0
        for index,key in enumerate(graphs.keys()):
            model=models[index]
            graph = graphs[key]
            weights = OrderedDict(model[0].named_parameters())
            if task == 'EdgePred':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)


            elif task == 'sim':
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)


                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)


                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])


            elif task == 'AttrMask':
                z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)
                z_val1 = model[1](z_val)
                loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

                z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                z_val2 = model[1](z_val)
                loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                          graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1


                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            cum+=graph.x.shape[0]
            loss+=(loss1+loss2)

        loss +=  2* fused_graph.adj_tensor.mean()
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()


    adj = match_graph(graphs)

    if task == 'dgi':

        original_adj = fused_graph.original_adj
        arr = fused_graph.arr

        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,graphs,0.99),True)
        fused_graph.original_adj = original_adj
        fused_graph.arr = arr
    else:
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,graphs,0.99),True)


    return fused_graph


def MDP_gradient_fuse_sim(args, task, pretrain_model, auxiliary, graphs,fugraphs, optimizer,modality_projections,wdiscriminator,optimizer_wd,match_graph, epoch, device, file):


    fused_graph = lambda_for_adj(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)


    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    graphs['fuse'] = fused_graph


    print(graphs['fuse'].x.shape)
    print(graphs['fuse'].edge_index.shape)

    weights = OrderedDict(pretrain_model.named_parameters())


    dis_losses = []

    for index,name in enumerate( graphs.keys()):

        Dis_loss = torch.tensor(0.0).to(device)
        graph= graphs[name]
        if name!='fuse':


            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, graphs['fuse'], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w1 = wdiscriminator_copy( graphs['fuse'].x, graphs['fuse'].edge_index)
            w0 = wdiscriminator_copy(graph.x, graph.edge_index)

            Dis_loss = (torch.mean(w1) - torch.mean(w0))

            dis_losses.append(Dis_loss)

            print(Dis_loss.item())


    file.write('{},{},{},'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in dis_losses])))
    file.flush()

    x2s = []
    x1s = []
    vice_model = gen_ran_output(pretrain_model)
    for index,name in enumerate( graphs.keys()):
        graph = graphs[name]
        embedding = vice_model.encoder(graph.x, graph.edge_index, weights).mean(0)
        x2s.append(auxiliary(embedding))
        x1s.append(auxiliary(pretrain_model.encoder(graph.x, graph.edge_index, weights).mean(0)))
    x2 = torch.stack(x2s)
    x1 = torch.stack(x1s)
    loss = loss_cal(x1,x2)

    print(loss)


    optimizer.zero_grad()

    pretrain_batch_loss = loss+torch.stack(dis_losses).mean()
    pretrain_batch_loss.backward()

    optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_fuse(args, task, pretrain_model, auxiliary, graphs,fugraphs, optimizer,modality_projections,wdiscriminator,optimizer_wd,match_graph, epoch, device, file):


    fused_graph = lambda_for_adj(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)


    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation(modality_projections[name],int(np.sum(args.dims[:index])))


    graphs['fuse'] = fused_graph
    if task == 'AttrMask':graphs['fuse'].original_x = graphs['fuse'].x.clone().detach()


    print(graphs['fuse'].x.shape)
    print(graphs['fuse'].edge_index.shape)


    weights = OrderedDict(pretrain_model.named_parameters())


    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):

        Dis_loss = torch.tensor(0.0).to(device)
        graph= graphs[name]
        if name!='fuse':


            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, graphs['fuse'], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w1 = wdiscriminator_copy( graphs['fuse'].x, graphs['fuse'].edge_index)
            w0 = wdiscriminator_copy(graph.x, graph.edge_index)

            Dis_loss = (torch.mean(w1) - torch.mean(w0))


        if task == 'EdgePred':
            if name == 'fuse':
                graph.link_split(0.005, 0.005)
            else:graph.link_split(0.2, 0.2)
            print('graph.train_edge_index,',graph.train_edge_index.shape)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)

        elif task == 'AttrMask':

            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            if name == 'fuse':

                for ind,k in enumerate(args.train_dataset):
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    z_val[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,0:int(np.sum(args.dims[:ind]))]=0
                    graph.original_x[ind*args.fuse_scale:(1+ind)*args.fuse_scale,args.dims[ind]+int(np.sum(args.dims[:ind])):]=0


                loss = torch.nn.MSELoss()(z_val[graph.mask], graph.original_x[graph.mask])
            else:loss = torch.nn.MSELoss()(z_val[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]], graph.original_x[graph.mask])


        elif task == 'sim':
            vice_model = gen_ran_output(pretrain_model)
            output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
            output_negative = vice_model.encode(graph.x, graph.edge_index, weights)
            if name == 'fuse': loss = loss_cal2(output_positive,output_negative,auxiliary)
            else:loss = loss_cal2(output_positive,output_negative,auxiliary)

        else:


            if name == 'fuse':
                arr = torch.arange(graph.x.shape[0])
                output_positive = pretrain_model.encode(graph.x, graph.original_adj, weights)
            else:
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)


                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)


            output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)
            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary


            if name == 'fuse': loss = 0.1*(torch.nn.BCEWithLogitsLoss()(positive_score, 
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score)))
            else:loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))


        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in task_losses])))
    file.flush()


    if len(task_losses) != 0:
        optimizer.zero_grad()
        stack = torch.stack(task_losses)
        pretrain_batch_loss = stack.mean()+torch.stack(dis_losses).mean()
        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def MDP_gradient_fuse_new_mlp(args, task, pretrain_model, auxiliary, graphs,fugraphs, optimizer,modality_projections,wdiscriminator,optimizer_wd,match_graph, epoch, device, file):


    fused_graph = lambda_for_adj_new(args, task, pretrain_model, auxiliary,fugraphs, match_graph,device)


    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    pretrain_model.train()
    if auxiliary: auxiliary.train()


    for index,name in enumerate(args.train_dataset):
        graphs[name] = graphs[name].transformation_mlp(modality_projections[name])


    graphs['fuse'] = fused_graph
    if task == 'AttrMask':graphs['fuse'].original_x = graphs['fuse'].x.clone().detach()


    print(graphs['fuse'].x.shape)
    print(graphs['fuse'].edge_index.shape)


    weights = OrderedDict(pretrain_model.named_parameters())


    task_losses = []
    dis_losses = []

    for index,name in enumerate( graphs.keys()):

        Dis_loss = torch.tensor(0.0).to(device)
        graph= graphs[name]
        if name!='fuse':


            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(graph, graphs['fuse'], wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w1 = wdiscriminator_copy( graphs['fuse'].x, graphs['fuse'].edge_index)
            w0 = wdiscriminator_copy(graph.x, graph.edge_index)

            Dis_loss = (torch.mean(w1) - torch.mean(w0))


        if task == 'EdgePred':
            if name != 'fuse':
                graph.link_split(0.2, 0.2)

            print('graph.train_edge_index,',graph.train_edge_index.shape)
            z_val = pretrain_model.encode(graph.x, graph.train_edge_index, weights)
            loss = pretrain_model.recon_loss(z_val, graph.test_edge_index)


        elif task == 'AttrMask':

            graph.attr_split(0.2)
            z_val = pretrain_model.encode(graph.x, graph.edge_index, weights)
            z_val = auxiliary(z_val)
            loss = torch.nn.MSELoss()(z_val[graph.mask], graph.x[graph.mask])


        elif task == 'sim':
            if name == 'fuse': 
                vice_model = gen_ran_output(pretrain_model)
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)[:graph.num]
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)[:graph.num]
                loss = loss_cal2(output_positive,output_negative,auxiliary)
            else:
                vice_model = gen_ran_output(pretrain_model)
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)
                loss = loss_cal2(output_positive,output_negative,auxiliary)

        else:


            if name == 'fuse':
                arr = torch.arange(graph.x.shape[0])
                output_positive = pretrain_model.encode(graph.x, graph.original_adj, weights)[:graph.num]
                output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)[:graph.num]
            else:
                output_positive = pretrain_model.encode(graph.x, graph.edge_index, weights)


                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = pretrain_model.encode(graph.x[arr], graph.edge_index, weights)


            summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
            discriminator_summary = auxiliary(summary_emb).T
            positive_score = output_positive @ discriminator_summary
            negative_score = output_negative @ discriminator_summary


            if name == 'fuse': loss = (torch.nn.BCEWithLogitsLoss()(positive_score, 
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score)))
            else:loss = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
            negative_score, torch.zeros_like(negative_score))


        dis_losses.append(Dis_loss)
        task_losses.append(loss)
        print(Dis_loss.item(),loss.item())

    file.write('{},{},{},{}\n'.format(epoch, '-'.join(args.train_dataset), ','.join([str(lo.item()) for lo in task_losses]),','.join([str(lo.item()) for lo in dis_losses[:4]])))
    file.flush()


    if len(task_losses) != 0:
        optimizer.zero_grad()
        stack = torch.stack(task_losses)
        pretrain_batch_loss = stack.mean() +torch.stack(dis_losses).mean()

        pretrain_batch_loss.backward()

        optimizer.step()

    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)

    return pretrain_model, pretrain_batch_loss.item()


def fused_split_new(fused_graph,graphs,task):


    print(' graphs.keys(),', graphs.keys())
    fused_graph.x = torch.cat([graphs[name].x.detach() for name in graphs.keys()]+[fused_graph.super_nodes],dim =0)


    if task == 'EdgePred':
        cum = 0
        test_index = []

        for key in graphs.keys():
            graph = graphs[key]

            graph.link_split(0.2, 0.2)
            test_index.append(cum+graph.test_edge_index)

            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.train_edge_index[0]+cum,graph.train_edge_index[1]+cum] = 1.0

            cum+=graph.x.shape[0]

        fused_graph.test_index = test_index
        fused_graph.adj_tensor = fused_graph.adj

    elif task == 'AttrMask':

        cum = 0
        test_index = []
        for key in graphs.keys():
            graph = graphs[key]
            graph.attr_split2(0.2)
            test_index.append(graph.mask+cum)
            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]


        fused_graph.test_index = test_index
        fused_graph.adj_tensor = fused_graph.adj

        fused_graph.xx = torch.cat([graphs[name].xx for name in graphs.keys()]+[fused_graph.super_nodes],dim =0)


    elif task == 'sim':


        cum = 0
        for key in graphs.keys():
            graph = graphs[key]
            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]

        fused_graph.adj_tensor = fused_graph.adj


    else:

        all_edge_index = []

        cum = 0

        for key in graphs.keys():
            graph = graphs[key]

            all_edge_index.append(graph.edge_index+cum)

            fused_graph.adj[cum:cum+graph.x.shape[0],cum:cum+graph.x.shape[0]] = 0.0
            fused_graph.adj[graph.edge_index[0]+cum,graph.edge_index[1]+cum] = 1.0
            cum+=graph.x.shape[0]


        fused_graph.adj_tensor = fused_graph.adj
        fused_graph.original_adj = torch.cat(all_edge_index,dim=1)


        tmp = []
        node_nums = np.cumsum([0]+[graphs[name].x.shape[0] for name in graphs.keys()])
        for i in range(len(node_nums)-1):
            tmp.append(np.arange(node_nums[i],node_nums[i+1])+1)
            tmp[-1][-1] = node_nums[i]
        tmp.append(np.arange(node_nums[-1],fused_graph.x.shape[0]))
        arr = np.concatenate(tmp)

        fused_graph.arr = arr
    return fused_graph,graphs


def lambda_for_adj_new(args, task, base_model, base_auxiliary,graphs,match_graph,device):


    optimizer= torch.optim.Adam(match_graph.parameters(), lr=0.001,weight_decay=5e-4)


    for loop in range(5):


        adj = match_graph(graphs)

        fused_graph = Graph(None,None)
        fused_graph.adj = adj
        fused_graph.super_nodes = match_graph.super_nodes
        fused_graph,graphs = fused_split_new(fused_graph,graphs,task)

        print(fused_graph.adj_tensor.sum(),fused_graph.adj_tensor)

        models = lambda_training_new(args, task, base_model, base_auxiliary, fused_graph,graphs,device)


        cum = 0
        loss = 0
        for index,key in enumerate(graphs.keys()):
            model=models[index]
            graph = graphs[key]
            weights = OrderedDict(model[0].named_parameters())
            if task == 'EdgePred':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor)

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)


            elif task == 'sim':
                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)


                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)


                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])


            elif task == 'AttrMask':

                if 'new_dims' in args:
                    z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                            weights = weights, edge_weight = fused_graph.adj_tensor)
                    z_val1 = model[1](z_val)
                    loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],args.old_dims[index][0]:args.old_dims[index][1]],
                                            graph.original_x[graph.mask,:])

                    z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                    z_val2 = model[1](z_val)
                    loss2 = torch.nn.MSELoss()(z_val2[graph.mask,args.old_dims[index][0]:args.old_dims[index][1]],
                                            graph.original_x[graph.mask,:])
                else:
                    z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                            weights = weights, edge_weight = fused_graph.adj_tensor)
                    z_val1 = model[1](z_val)
                    loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

                    z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                    z_val2 = model[1](z_val)
                    loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1


                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor)
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))

            cum+=graph.x.shape[0]
            loss+=(loss1+loss2)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()


    adj = match_graph(graphs)


    if task == 'dgi':
        original_adj = fused_graph.original_adj
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,args.tau),True)
        fused_graph.original_adj = original_adj


    elif task == 'EdgePred':


        adj_tensor = fused_graph.adj_tensor
        test_index = torch.cat(fused_graph.test_index,1)
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,args.tau),True)
        fused_graph.test_edge_index = test_index
        adj[:adj.shape[0]-match_graph.proto_num,:adj.shape[0]-match_graph.proto_num]=adj_tensor[:adj.shape[0]-match_graph.proto_num,:adj.shape[0]-match_graph.proto_num]
        fused_graph.train_edge_index = match_graph.get_final_adj(adj,args.tau)
    else:
        fused_graph = Graph(fused_graph.x,match_graph.get_final_adj(adj,args.tau),True)


    fused_graph.num = fused_graph.x.shape[0]-match_graph.proto_num
    if (fused_graph.edge_index>=fused_graph.num).sum()==0:
        print('000000')
        fused_graph.x = fused_graph.x[:fused_graph.num ]
    return fused_graph


def lambda_training_new(args, task, base_model, base_auxiliary, fused_graph,graphs,device):

    models = []

    cum = 0
    for index,key in enumerate(graphs.keys()):
        graph = graphs[key]
        model = deepcopy(base_model),deepcopy(base_auxiliary)
        model_params = []
        model_params.extend(model[0].parameters())
        if model[1] is not None: model_params.extend(model[1].parameters())


        optimizer= torch.optim.Adam([{'params': model_params}], lr=0.001,weight_decay=5e-4)   

        for epoch in range(10):
            model[0].train()
            if model[1] is not None:model[1].train()

            weights = OrderedDict(model[0].named_parameters())

            if task == 'EdgePred':
                z_val1 = model[0].encode(x = fused_graph.x, edge_index = None,
                                           weights = weights, edge_weight = fused_graph.adj_tensor.detach())

                loss1 = model[0].recon_loss(z_val1, fused_graph.test_index[index])

                z_val2 = model[0].encode(graph.x.detach(), graph.train_edge_index, weights)
                loss2 = model[0].recon_loss(z_val2, graph.test_edge_index)

            elif task == 'sim':


                vice_model = gen_ran_output(model[0])

                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())


                output_negative = vice_model.encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())


                loss1 = loss_cal2(output_positive,output_negative,model[1])


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                output_negative = vice_model.encode(graph.x, graph.edge_index, weights)

                loss2 = loss_cal2(output_positive,output_negative,model[1])


            elif task == 'AttrMask':

                if 'new_dims' in args:
                    z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                            weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                    z_val1 = model[1](z_val)
                    loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],args.old_dims[index][0]:args.old_dims[index][1]],
                                            graph.original_x[graph.mask,:])

                    z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                    z_val2 = model[1](z_val)
                    loss2 = torch.nn.MSELoss()(z_val2[graph.mask,args.old_dims[index][0]:args.old_dims[index][1]],
                                            graph.original_x[graph.mask,:])


                else:


                    z_val = model[0].encode(x = fused_graph.xx, edge_index = None,
                                            weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                    z_val1 = model[1](z_val)
                    loss1 = torch.nn.MSELoss()(z_val1[fused_graph.test_index[index],int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

                    z_val = model[0].encode(graph.xx, graph.edge_index, weights)
                    z_val2 = model[1](z_val)
                    loss2 = torch.nn.MSELoss()(z_val2[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]],
                                            graph.x[graph.mask,int(np.sum(args.dims[:index])):int(np.sum(args.dims[:index]))+args.dims[index]])

            else:
                embedding = fused_graph.x
                output_positive = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_positive = output_positive[cum:cum+graph.x.shape[0],:]

                arr = np.arange(embedding.shape[0])
                arr[cum+graph.x.shape[0]-1]= arr[cum]-1
                arr[cum:cum+graph.x.shape[0]] = arr[cum:cum+graph.x.shape[0]] +1


                embedding = embedding[arr]

                output_negative = model[0].encode(x = embedding, edge_index = None,weights = weights, edge_weight = fused_graph.adj_tensor.detach())
                output_negative = output_negative[cum:cum+graph.x.shape[0],:]
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss1 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


                output_positive = model[0].encode(graph.x, graph.edge_index, weights)
                arr = torch.arange(graph.x.shape[0]) + 1
                arr[-1:] = torch.arange(1)
                output_negative = model[0].encode(graph.x[arr], graph.edge_index, weights)
                summary_emb = torch.sigmoid(torch.mean(output_positive, dim=0, keepdim=True))
                discriminator_summary = model[1](summary_emb).T
                positive_score = output_positive @ discriminator_summary
                negative_score = output_negative @ discriminator_summary
                loss2 = torch.nn.BCEWithLogitsLoss()(positive_score,
                                                torch.ones_like(positive_score)) + torch.nn.BCEWithLogitsLoss()(
                negative_score, torch.zeros_like(negative_score))


            loss = loss1+loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(loss1,loss2)
        cum+=graph.x.shape[0]


        for p in model[0].parameters(): p.requires_grad = False
        if model[1] is not None:
            for p in model[1].parameters(): p.requires_grad = False
        models.append(model)


    return models


def construct_challenging_masks(args, subgraphs, view_constructor, pretrain_model, modality_projections, auxiliary,
                  semantic_space=None, optimizer=None):
    return optimize_view_constructor(args, subgraphs, view_constructor, pretrain_model, modality_projections,
                            auxiliary, semantic_space, optimizer)


def construct_challenging_masks_for_edges(args, subgraphs, view_constructor, pretrain_model, modality_projections,
                       semantic_space=None, optimizer=None):
    return optimize_view_constructor(args, subgraphs, view_constructor, pretrain_model, modality_projections,
                            {}, semantic_space, optimizer)


def weighted_semantic_grounding(node_features, semantic_factors, top_k=5, eta=0.25):
    """
    Semantic grounding with similarity-weighted top-K factor selection.

    Args:
        node_features (torch.Tensor): Node features of shape [N, D], with N nodes and D feature dimensions.
        semantic_factors (torch.Tensor): Semantic factors of shape [T, D], with T factors and D feature dimensions.
        top_k (int): Number of semantic factors selected for each node.
        eta (float): Weight of the commitment loss term.

    Returns:
        loss (torch.Tensor): Weighted semantic grounding loss.
    """

    # Compute node-to-factor similarities.


    similarity = torch.matmul(node_features, semantic_factors.T)  # Shape [N, T].

    # Select the top-K semantic factor indices by similarity for each node.
    top_k_sim, top_k_indices = similarity.topk(top_k, dim=-1, largest=True, sorted=False)  # Shape [N, k].
    top_k_sim = torch.softmax(top_k_sim, dim=-1)
    node_features_stop = node_features.detach()  # stop-gradient
    selected_factors = semantic_factors[top_k_indices]


    factor_alignment_loss = torch.mean(torch.sum(top_k_sim*torch.norm(node_features_stop.unsqueeze(1)- selected_factors, p=2, dim=-1), dim=-1))  # [N]


    # Stop-gradient on the semantic factors in the commitment term.

    detached_factors = semantic_factors.detach()  # stop-gradient
    detached_selected_factors = detached_factors[top_k_indices]
    commitment_loss = torch.mean(torch.sum(top_k_sim*torch.norm(node_features.unsqueeze(1)- detached_selected_factors, p=2, dim=-1), dim=-1))

    # Combine factor alignment with the weighted commitment loss.
    loss = factor_alignment_loss + eta * commitment_loss

    return loss


def pretrain_epoch(args, pretrain_model, semantic_space, view_constructor, modality_projections, auxiliary,
                subgraphs, optimizer, epoch, device, file=None, constructor_optimizer=None):
    pretrain_model.train(); semantic_space.train()
    for module in list(modality_projections.values())+list(auxiliary.values()):
        module.train()
    if not subgraphs:
        raise ValueError('An epoch needs at least one sampled subgraph')
    masks = optimize_view_constructor(args, subgraphs, view_constructor, pretrain_model, modality_projections,
                              auxiliary, semantic_space, constructor_optimizer)
    optimizer.zero_grad(set_to_none=True)
    totals = np.zeros(4, dtype=float)
    for graph, challenging_mask in zip(subgraphs, masks):
        rate = np.random.uniform(args.random_missing_min, args.random_missing_max)
        random_mask = make_modality_mask(graph.text_feat.size(0), 3, rate,
                          seed=int(np.random.randint(0, 2**31-1))).to(device)
        challenging_embeddings, challenging_grounding = encode_pass(pretrain_model, semantic_space, modality_projections, graph, challenging_mask)
        challenging_loss = pretraining_loss(args, pretrain_model, challenging_embeddings, auxiliary, graph, challenging_mask)
        random_embeddings, random_grounding = encode_pass(pretrain_model, semantic_space, modality_projections, graph, random_mask)
        random_loss = pretraining_loss(args, pretrain_model, random_embeddings, auxiliary, graph, random_mask)
        grounding = challenging_grounding + random_grounding
        # Joint pre-training objective, Eq. (17).
        loss = challenging_loss + args.lambda1*random_loss + args.lambda2*grounding
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite pre-training loss at epoch {epoch}')
        (loss/len(subgraphs)).backward()
        totals += [challenging_loss.item(), random_loss.item(), grounding.item(), loss.item()]
    optimizer.step()
    totals /= len(subgraphs)
    if file is not None:
        file.write(','.join(map(str, [epoch]+totals.tolist()))+'\n')
        file.flush()
    print(f'epoch={epoch} challenging={totals[0]:.6f} random={totals[1]:.6f} '
          f'grounding={totals[2]:.6f} total={totals[3]:.6f}')
    return pretrain_model, semantic_space, float(totals[3])


def encode_pass(model, semantic_space, modality_projections, graph, mask, edge_index=None,
                calibration_adapter=None, compute_grounding=True):
    """Shared encoding path for constructor, pre-training and adaptation."""
    edges = graph.edge_index if edge_index is None else edge_index
    raw = [graph.text_feat, graph.vision_feat, graph.structure_feat]
    features = [modality_projections[modal](raw[m]*mask[:, m:m+1])
                for m, modal in enumerate(('text','vision','structure'))]
    factors = semantic_space(graph, mask, edge_index=edges, calibration_adapter=calibration_adapter)
    _, _, embeddings, _ = model.encode(features, edges, factors, mask=mask,
                                       compute_grounding=compute_grounding)
    return embeddings, model.encoder.last_grounding_loss


def pretraining_loss(args, model, embeddings, auxiliary, graph, mask):
    if args.pretrain_task == 'EdgePred':
        return sum(model.recon_loss(z, graph.target_edge_index,
                                   neg_edge_index=graph.target_negative_edge_index)
                   for z in embeddings)
    # Attribute reconstruction over masked observations.

    raw = [graph.text_feat, graph.vision_feat, graph.structure_feat]
    loss = embeddings[0].sum()*0.0
    for m, modal in enumerate(('text','vision','structure')):
        error = (auxiliary[modal](embeddings[m])-raw[m]).square().mean(dim=-1)
        missing = 1-mask[:, m]
        loss = loss + (error*missing).sum()/missing.sum().clamp_min(1)
    return loss


def optimize_view_constructor(args, graphs, view_constructor, pretrain_model, modality_projections,
                     auxiliary, semantic_space, optimizer):
    """Eqs. (2)--(4): J ascent steps through the complete frozen PASS model."""
    if semantic_space is None:
        raise ValueError('Challenge construction requires the full PASS semantic module')
    if optimizer is None:
        optimizer = torch.optim.Adam(view_constructor.parameters(), lr=args.constructor_lr)
    modules = [pretrain_model, semantic_space]+list(modality_projections.values())+list(auxiliary.values())
    params = list({id(p): p for module in modules for p in module.parameters()}.values())
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    view_constructor.train()
    try:
        for _ in range(args.constructor_ascent_steps):
            optimizer.zero_grad(set_to_none=True)
            for graph in graphs:
                features = torch.cat((graph.text_feat, graph.vision_feat, graph.structure_feat), dim=1)
                logits = view_constructor(features, graph.edge_index)
                mask = F.gumbel_softmax(logits, tau=args.tau, hard=True, dim=-1)
                embeddings, _ = encode_pass(pretrain_model, semantic_space, modality_projections, graph, mask,
                                             compute_grounding=False)
                # Maximize the pre-training objective, Eq. (4).
                loss = pretraining_loss(args, pretrain_model, embeddings, auxiliary, graph, mask)
                (-loss/len(graphs)).backward()
            optimizer.step()
        # Constructor is fixed in the subsequent encoder step.
        with torch.no_grad():
            masks = []
            for graph in graphs:
                features = torch.cat((graph.text_feat, graph.vision_feat, graph.structure_feat), dim=1)
                masks.append(F.gumbel_softmax(view_constructor(features, graph.edge_index),
                                             tau=args.tau, hard=True, dim=-1).detach())
        return masks
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def prepare_pretraining_targets(graph, task, edge_mask_rate=0.2):
    """Construct EdgePred reconstruction targets and the message-passing graph."""
    if task != 'EdgePred':
        return
    from torch_geometric.utils import negative_sampling as sample_negative
    full = graph.edge_index
    canonical = full[:, full[0] < full[1]]
    if canonical.size(1) < 2:
        raise ValueError('Sampled graph has too few edges for EdgePred')
    order = torch.randperm(canonical.size(1), device=canonical.device)
    count = max(1, min(canonical.size(1)-1, int(canonical.size(1)*edge_mask_rate)))
    graph.target_edge_index = canonical[:, order[:count]]
    graph.full_edge_index = full
    graph.edge_index = to_undirected(canonical[:, order[count:]], num_nodes=graph.text_feat.size(0))
    graph.target_negative_edge_index = sample_negative(
        full, num_nodes=graph.text_feat.size(0), num_neg_samples=count, method='sparse')
    if graph.target_negative_edge_index.numel() == 0:
        raise ValueError('No negative targets in sampled graph')


@torch.no_grad()
def evaluate_graph(modality_projections, semantic_space, model, graph, calibration_adapter, split='test'):
    model.eval(); semantic_space.eval(); calibration_adapter.eval()
    for module in modality_projections.values():
        module.eval()
    embeddings, _ = encode_pass(model, semantic_space, modality_projections, graph, graph.mask,
                                 calibration_adapter=calibration_adapter, compute_grounding=False)
    embedding = torch.cat(embeddings, dim=1)
    if model.task == 'link':
        embedding = model.decoder.mlp(embedding)
        return float(model.test(embedding, getattr(graph, split+'_edge_index'),
                                getattr(graph, split+'_edge_index_negative'))[0])
    index = getattr(graph, split+'_labels')
    return float(model.class_test(embedding[index], graph.labels[index]))

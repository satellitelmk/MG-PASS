from collections import defaultdict
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os.path
from pathlib import Path
from random import randint
import random
import math
from torch_geometric.utils import to_undirected,subgraph
from models.models import MLP


def negative_sampling_identify(pos_edge_index, num_nodes):

    def fetch(arr,num,r = 0):
        d = int((len(arr)+0.0)/num)
        result = list(range(r,len(arr),d))
        if len(result)>num:result = result[:num]
        return result


    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1]) # Flatten edge coordinates.
    idx = idx.to(torch.device('cpu'))

    rng =  [i * num_nodes+j for i in range(num_nodes) for j in range(num_nodes) if i<j ]
    perm = torch.tensor(fetch(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero(as_tuple = False).view(-1)

    index = 0
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(fetch(rng, rest.size(0),index))
        perm[rest] = tmp
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero(as_tuple = False).view(-1)
        index+=1

    row, col = torch.div(perm, num_nodes,rounding_mode='floor'), perm % num_nodes   


    return torch.stack([row, col], dim=0).to(pos_edge_index.device)


def load_test_graph(dataset, index, text, vision, structure, norm=False,
                    data_root='dataset', train_ratio=0.01, seed=0):
    """Load a cached subgraph, its modality features and supervision split."""
    root = Path(data_root) / dataset
    stem = split_stem(index, train_ratio, seed)
    split_path = root / 'splits' / (stem+'.npz')
    if not split_path.exists():
        raise FileNotFoundError(f'{split_path}: run prepare_dataset.py once first')
    with np.load(split_path, allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in saved.files}
    with np.load(root/'test_graphs'/f'test_graph{index}.npz', allow_pickle=True) as saved:
        sample = np.asarray(saved['sample'], dtype=np.int64)
        if not np.array_equal(sample, split['sample']):
            raise ValueError('Cached split belongs to different sampled nodes')
        feats = {}
        for modal, name in [('text', text), ('vision', vision)]:
            key = 'feats_'+modal
            if key in saved:
                value = saved[key]
                feats[modal] = value.item()[name] if value.dtype == object else value
            else:
                feats[modal] = np.load(root/f'{name}_{modal}_feature_raw.npy', mmap_mode='r')[sample]
        labels = saved['labels'] if 'labels' in saved else None
        if labels is not None and labels.size == 0:
            labels = None
    task = str(split['task'].item())
    structures = load_structure_features(root, structure, task, stem, split)
    feats['structure'] = structures[sample]


    if norm:
        raise ValueError('Use observed-only preprocessing; full-feature scaling is not enabled')
    graph = test_graph(feats['text'], feats['vision'], feats['structure'],
                       split['message_edge_index'], labels=labels)
    graph.sample = sample
    graph.full_edge_index = split['full_edge_index']
    graph.cached_split = True
    graph.split_ratio = train_ratio
    graph.split_stem = stem
    graph.split_digest = str(split['split_digest'].item())
    graph.task = task
    graph.modal_names = ['text', 'vision', 'structure']
    for key in ('train_labels','val_labels','test_labels','train_edge_index',
                'val_edge_index','test_edge_index','train_edge_index_negative',
                'val_edge_index_negative','test_edge_index_negative'):
        if key in split:
            setattr(graph, key, split[key])
    return graph


def load_test_graph_share(dataset,index,num):
    dataset_str = '/home/lmk/LAMP-GNN/lamp_data/test/' + dataset + '/'
    file = dataset_str+'test_graph{}.npz'.format(index)
    file = np.load(file)
    feats = file['feats']

    mlp = MLP(feats.shape[1], 128, 128)
    mlp.load_state_dict(torch.load('./result/autoencoder_for_{}_{}.pth'.format(dataset,num)), strict=True)


    feats = mlp(torch.from_numpy(feats).to(torch.float)).cpu().detach().numpy()

    scaler = StandardScaler()
    scaler.fit(feats)
    feats = scaler.transform(feats)

    edge_index = file['edge_index']
    labels = file['labels']
    graph = Graph(x = feats,edge_index=edge_index,is_tensor=False,labels=labels)
    return graph


class test_graph:
    def __init__(self,text_feat,vision_feat,structure_feat,edge_index,labels = None,is_tensor = False):
        super(test_graph, self).__init__()
        self.text_feat = text_feat
        self.vision_feat = vision_feat
        self.structure_feat = structure_feat
        self.edge_index = edge_index
        self.labels = labels
        self.is_tensor = is_tensor


    def link_split(self,train_ratio,test_ratio):

        row, col = self.edge_index


    # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(train_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        train_pos_edge_index = torch.stack([r, c], dim=0)
        train_pos_edge_index = to_undirected(train_pos_edge_index)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        val_pos_edge_index = torch.stack([r, c], dim=0)


        train_pos_edge_index = train_pos_edge_index.detach()
        val_pos_edge_index = val_pos_edge_index.detach()
        test_pos_edge_index = test_pos_edge_index.detach()

        num_nodes = self.text_feat.shape[0]
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)

        self.train_edge_index = train_pos_edge_index
        self.test_edge_index = test_pos_edge_index

    def get_sub_edge_index(self,edge_index, list_sample, relabel_nodes=True):
        """
        Extract the subgraph induced by the nodes in list_sample.
        
        Args:
            edge_index (LongTensor): Edge indices of the original graph, shape [2, E].
            list_sample (list or LongTensor): Node IDs to retain.
            relabel_nodes (bool): Whether to relabel subgraph nodes with consecutive IDs starting at zero.
        
        Returns:
            sub_edge_index (LongTensor): Edge indices of the subgraph.
            node_mapping (dict): Mapping from original to local node IDs when relabel_nodes=True.
        """
        if not torch.is_tensor(list_sample):
            list_sample = torch.tensor(list_sample, dtype=torch.long)

        sub_edge_index, _ = subgraph(
            subset=list_sample,
            edge_index=edge_index,
            relabel_nodes=relabel_nodes
        )

        node_mapping = None
        if relabel_nodes:
            node_mapping = list_sample

        return sub_edge_index, node_mapping

    def get_concat_graph(self,device):
        text_feat = np.zeros_like(self.text_feat)
        vision_feat = np.zeros_like(self.vision_feat)
        structure_feat = np.zeros_like(self.structure_feat)

        text_feat[self.text_list] = self.text_feat[self.text_list]
        vision_feat[self.vision_list] = self.vision_feat[self.vision_list]
        structure_feat[self.structure_list] = self.structure_feat[self.structure_list]

        feats = np.concatenate([text_feat,vision_feat,structure_feat],axis = 1)
        return Graph(feats,self.edge_index,False,self.labels)


    def get_incomplete_features(self,):
        text_feat = np.zeros_like(self.text_feat)
        vision_feat = np.zeros_like(self.vision_feat)
        structure_feat = np.zeros_like(self.structure_feat)

        text_feat[self.text_list] = self.text_feat[self.text_list]
        vision_feat[self.vision_list] = self.vision_feat[self.vision_list]
        structure_feat[self.structure_list] = self.structure_feat[self.structure_list]

        self.text_feat = text_feat
        self.vision_feat = vision_feat
        self.structure_feat = structure_feat


    def transformation(self,modality_projections):

        return test_graph(text_feat=modality_projections['text'](self.text_feat),vision_feat=modality_projections['vision'](self.vision_feat),structure_feat=modality_projections['structure'](self.structure_feat),edge_index = self.edge_index)


    def to_tensor(self, device):
        self.device = device
        self.text_feat = torch.as_tensor(self.text_feat, dtype=torch.float32, device=device)
        self.vision_feat = torch.as_tensor(self.vision_feat, dtype=torch.float32, device=device)
        self.structure_feat = torch.as_tensor(self.structure_feat, dtype=torch.float32, device=device)
        for key in ('edge_index','full_edge_index','train_edge_index','val_edge_index',
                    'test_edge_index','train_edge_index_negative','val_edge_index_negative',
                    'test_edge_index_negative','train_labels','val_labels','test_labels',
                    'target_edge_index','target_negative_edge_index'):
            if hasattr(self, key):
                setattr(self, key, torch.as_tensor(getattr(self, key), dtype=torch.long, device=device))
        if self.labels is not None:
            dtype = torch.float32 if len(self.labels.shape) == 2 else torch.long
            self.labels = torch.as_tensor(self.labels, dtype=dtype, device=device)
        self.is_tensor = True
        return self


    def divide_graph(self,device):

        edge_index_text, self.node_mapping_text = self.get_sub_edge_index(self.edge_index,self.text_list,True)
        self.graph_text = Graph(self.text_feat[self.text_list],edge_index_text).to(device)

        edge_index_vision, self.node_mapping_vision = self.get_sub_edge_index(self.edge_index,self.vision_list,True)
        self.graph_vision = Graph(self.vision_feat[self.vision_list],edge_index_vision).to(device)


        edge_index_structure, self.node_mapping_structure = self.get_sub_edge_index(self.edge_index,self.structure_list,True)
        self.graph_structure = Graph(self.structure_feat[self.structure_list],edge_index_structure).to(device)


    def complete_modalities_graph(self,text_feats, vision_feats, structure_feats):
        """
        Complete missing modalities using vectorized operations.
        text_feats, vision_feats, structure_feats: (N, d) torch.FloatTensor
        text_list, vision_list, structure_list: list[int]
        Returns: Completed text, vision, and structure features.
        """
        N, d = text_feats.shape

        text_list, vision_list, structure_list = self.text_list,self.vision_list,self.structure_list

        # Build the observation mask for the three modalities, shape (N, 3).
        mask = torch.zeros((N, 3), dtype=torch.float32, device=text_feats.device)
        mask[text_list, 0] = 1
        mask[vision_list, 1] = 1
        mask[structure_list, 2] = 1


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

        return {'text':text_feats_new,'vision':vision_feats_new,'structure':structure_feats_new} 


    def multimodal_to_isolate_graph(self,convert_text,convert_vision,convert_structure ):

        text_set = set(self.text_list)
        vision_set = set(self.vision_list)
        structure_set = set(self.structure_list)
        N, d = convert_text.shape


        texts,visions,structures = [],[],[]
        for i in range(N):
            has_text = i in text_set
            has_vision = i in vision_set
            has_structure = i in structure_set

            if has_text + has_vision + has_structure == 1:
                # Use the single observed modality to fill the other two.
                if has_text:
                    visions.append(convert_text[i])
                    structures.append(convert_text[i])
                    texts.append(convert_text[i])
                elif has_vision:
                    texts.append(convert_vision[i])
                    structures.append(convert_vision[i])
                    visions.append(convert_vision[i])
                elif has_structure:
                    texts.append(convert_structure[i])
                    visions.append(convert_structure[i])
                    structures.append(convert_structure[i])

            elif has_text + has_vision + has_structure == 2:
                # Fill the missing modality with the mean of the other two.
                if not has_text:
                    texts.append((convert_vision[i] + convert_structure[i]) / 2)
                    visions.append(convert_vision[i])
                    structures.append(convert_structure[i])

                if not has_vision:
                    visions.append((convert_text[i] + convert_structure[i]) / 2)
                    structures.append(convert_structure[i])
                    texts.append(convert_text[i])
                if not has_structure:
                    structures.append((convert_text[i] + convert_vision[i]) / 2)
                    texts.append(convert_text[i])
                    visions.append(convert_vision[i])

        text_feats = torch.stack(texts,dim=0)
        vision_feats = torch.stack(visions,dim=0)
        structure_feats = torch.stack(structures,dim=0)


        return [text_feats,vision_feats,structure_feats]


    def multimodal_to_flat_graph(self,task ):
        """
        Expand a multimodal graph into a PyG graph for node classification or link prediction.

        Args:
            edge_index: (2, E) edges of the original graph.
            text_feats, vision_feats, structure_feats: (N, d) features for each modality.
            text_list, vision_list, structure_list: list[int] of nodes with the corresponding modality.
            labels: (N,) original node labels.
            task: "node_cls" | "link_pred"
            train_idx, val_idx, test_idx: list[int] (node classification only).
            train_edge_index, val_edge_index, test_edge_index, test_edge_index_negative: (2, M) tensor (link prediction only).

        Returns:
            new_data: PyG Data object.
            extra_info: Dictionary of modality mappings for training, validation, and testing.
        """

        device = self.edge_index.device
        edge_index = self.edge_index
        text_feats = self.text_feat
        vision_feats = self.vision_feat
        structure_feats = self.structure_feat
        text_list = self.text_list
        vision_list = self.vision_list
        structure_list = self.structure_list
        labels = self.labels
        if task == 'node':
            train_idx = self.train_labels
            val_idx = self.val_labels
            test_idx = self.test_labels
        else:
            train_edge_index = self.train_edge_index
            val_edge_index = self.val_edge_index
            test_edge_index = self.test_edge_index
            test_edge_index_negative = self.test_edge_index_negative


        N, d = text_feats.shape
        feats = []
        modality_map = {}  # (node, modality) -> new_node_id
        node2modalities = {}  # node_id -> [new_node_ids]
        cur_id = 0

        # Create a node for each observed modality.
        for i in range(N):
            node2modalities[i] = []
            if i in text_list:
                feats.append(text_feats[i])
                modality_map[(i, "text")] = cur_id
                node2modalities[i].append(cur_id)
                cur_id += 1
            if i in vision_list:
                feats.append(vision_feats[i])
                modality_map[(i, "vision")] = cur_id
                node2modalities[i].append(cur_id)
                cur_id += 1
            if i in structure_list:
                feats.append(structure_feats[i])
                modality_map[(i, "structure")] = cur_id
                node2modalities[i].append(cur_id)
                cur_id += 1

        feats = torch.stack(feats, dim=0).to(device)

        edge_set = set()


        for i in range(N):
            mods = node2modalities[i]
            for j in range(len(mods)):
                for k in range(j+1, len(mods)):
                    u, v = mods[j], mods[k]
                    edge_set.add((u, v))
                    edge_set.add((v, u))


        E = edge_index.shape[1]
        for e in range(E):
            u, v = edge_index[0, e].item(), edge_index[1, e].item()
            for mu in node2modalities[u]:
                for mv in node2modalities[v]:
                    edge_set.add((mu, mv))
                    edge_set.add((mv, mu))

        edge_index_new = torch.LongTensor(list(edge_set)).t().contiguous().to(device)

        new_data = Graph(x=feats, edge_index=edge_index_new,is_tensor=True,labels=labels)

        new_data.original_feats =self.original_feats
        new_data.modal_lists = {'text':text_list,'vision':vision_list,'structure':structure_list}


        extra_info = {}

        if task == "node":
            # Expand training nodes into modality nodes.
            train_nodes = [[],[]]

            for i in train_idx:
                train_nodes[0].extend(node2modalities[i])
                train_nodes[1].extend([labels[i]]*len(node2modalities[i]))
            extra_info["train_idx"] =[np.array(train_nodes[0]),torch.LongTensor(train_nodes[1]).to(device)]

            # Keep modality-node groups for each original validation or test node.
            val_nodes = [[],[],[]]
            for index,i in enumerate(val_idx):
                val_nodes[0].extend([index]*len(node2modalities[i]))
                val_nodes[1].extend(node2modalities[i])
                val_nodes[2].append(labels[i])
            extra_info["val_idx"] =[torch.LongTensor(val_nodes[0]).to(device),np.array(val_nodes[1]),torch.LongTensor(val_nodes[2]).to(device)]

            test_nodes = [[],[],[]]
            for index,i in enumerate(test_idx):
                test_nodes[0].extend([index]*len(node2modalities[i]))
                test_nodes[1].extend(node2modalities[i])
                test_nodes[2].append(labels[i])


            extra_info["test_idx"] =[torch.LongTensor(test_nodes[0]).to(device),np.array(test_nodes[1]),torch.LongTensor(test_nodes[2]).to(device)]


        elif task == "link":
            # Expand training edges into modality edges.
            train_edges = []
            for e in range(train_edge_index.shape[1]):
                u, v = train_edge_index[0, e].item(), train_edge_index[1, e].item()
                train_edges = [(mu, mv) for mu in node2modalities[u] for mv in node2modalities[v]]
            for i in range(N):
                mods = node2modalities[i]
                for j in range(len(mods)):
                    for k in range(j+1, len(mods)):
                        u, v = mods[j], mods[k]
                        train_edges.append((u, v))
                        train_edges.append((v, u))
            extra_info["train_edge_index"] = torch.LongTensor(train_edges).t().contiguous().to(device)

            # Keep modality-edge groups for each original validation or test edge.
            def edge_map(edge_index,device):
                res = [[],[]]
                for e in range(edge_index.shape[1]):
                    u, v = edge_index[0, e].item(), edge_index[1, e].item()
                    edge_list = [(mu, mv) for mu in node2modalities[u] for mv in node2modalities[v]]
                    res[0].extend([e]*len(edge_list))
                    res[1].extend(edge_list)
                return [torch.LongTensor(res[0]).to(device),torch.LongTensor(res[1]).t().contiguous().to(device)]

            extra_info["val_map"] = edge_map(val_edge_index,device)
            extra_info["test_map"] = edge_map(test_edge_index,device)
            extra_info["test_neg_map"] = edge_map(test_edge_index_negative,device)
        new_data.extra_info = extra_info

        return new_data


    def aggregate_values(self,embeddings_list):
        emb_dim = embeddings_list.size(1)
        sum_emb = torch.zeros((self.text_feat.shape[0] , emb_dim), dtype=torch.float)
        count_emb = torch.zeros((self.text_feat.shape[0],), dtype=torch.float)
        for value,index in zip(embeddings_list[:len(self.node_mapping_text)],self.node_mapping_text):
            sum_emb[index]+=value
            count_emb[index] += 1
        for value,index in zip(embeddings_list[len(self.node_mapping_text):len(self.node_mapping_text)+len(self.node_mapping_vision)],self.node_mapping_vision):
            sum_emb[index]+=value
            count_emb[index] += 1
        for value,index in zip(embeddings_list[len(self.node_mapping_text)+len(self.node_mapping_vision):len(self.node_mapping_text)+len(self.node_mapping_vision)+len(self.node_mapping_structure)],self.node_mapping_structure):
            sum_emb[index]+=value
            count_emb[index] += 1
        final_emb = sum_emb / count_emb.unsqueeze(1)

        return final_emb


    def split_modal_nodes_degree(self, modal_ratio, seed=0, **kwargs):
        from utils import make_modality_mask
        return make_modality_mask(self.text_feat.shape[0], 3, 1-modal_ratio,
                                  'degree', self.edge_index, seed)


    def split_modal_nodes_snowball(self,
        modal_ratio,
        num_modalities = 3,
        domain_strength=1.0,
        max_hop=4
    ):
        """
        Returns:
            modality_node_lists: List[List[int]]
            Lists of nodes with observed features for each modality.
        """


        from collections import deque


        def build_adj_list_sorted(edge_index, num_nodes):
            adj = [[] for _ in range(num_nodes)]
            src, dst = edge_index

            for i in range(len(src)):
                u = src[i].item()
                v = dst[i].item()
                adj[u].append(v)
                adj[v].append(u)  # assume undirected

            # Sort neighbors for deterministic traversal.
            for i in range(num_nodes):
                adj[i] = sorted(adj[i])

            return adj

        def deterministic_two_hop_bfs(seed, adj, max_hop=2):
            visited = set([seed])
            queue = deque([(seed, 0)])
            result = []

            while queue:
                node, depth = queue.popleft()
                result.append(node)

                if depth >= max_hop:
                    continue

                for nei in adj[node]:
                    if nei not in visited:
                        visited.add(nei)
                        queue.append((nei, depth + 1))

            return result

        missing_rate = 1-modal_ratio

        num_nodes = self.text_feat.shape[0]
        edge_index = self.edge_index

        assert missing_rate <= (num_modalities - 1) / num_modalities, \
        "missing_rate too large for guaranteeing >=1 modality per node"

        adj = build_adj_list_sorted(edge_index, num_nodes)

        # Initialize all modalities as observed.
        modality_present = [set(range(num_nodes)) for _ in range(num_modalities)]
        node_modality_count = [num_modalities] * num_nodes

        target_missing = int(num_nodes * missing_rate)
        removed_count = [0] * num_modalities

        # Precompute a deterministic snowball traversal for each modality.

# Generate dispersed snowball traversal orders.


        snowball_order = []

        for m in range(num_modalities):

            order = []
            visited = set()

            # Estimate the number of clusters needed for each modality.
            cluster_size_est = len(deterministic_two_hop_bfs(0, adj, max_hop))
            approx_clusters = max(1, target_missing // max(1, cluster_size_est))

            stride = max(1, num_nodes // approx_clusters)

            # Select seeds at evenly spaced node indices.
            seeds = list(range(m, num_nodes, stride))

            for seed in seeds:
                cluster = deterministic_two_hop_bfs(seed, adj, max_hop)

                for node in cluster:
                    if node not in visited:
                        visited.add(node)
                        order.append(node)

            # Extend the traversal to cover any remaining nodes.
            if len(order) < num_nodes:
                for seed in range(num_nodes):
                    cluster = deterministic_two_hop_bfs(seed, adj, max_hop)
                    for node in cluster:
                        if node not in visited:
                            visited.add(node)
                            order.append(node)

            snowball_order.append(order)

        # Remove observations in round-robin order across modalities.
        progress = True
        while progress:
            progress = False

            for m in range(num_modalities):

                if removed_count[m] >= target_missing:
                    continue

                for node in snowball_order[m]:

                    if node not in modality_present[m]:
                        continue

                    if node_modality_count[node] > 1:
                        # Remove this modality observation.
                        modality_present[m].remove(node)
                        node_modality_count[node] -= 1
                        removed_count[m] += 1
                        progress = True
                        break

        # Validate the observation budget.
        for node in range(num_nodes):
            assert node_modality_count[node] >= 1

        self.text_list = sorted(list(modality_present[0]))
        self.vision_list = sorted(list(modality_present[1]))
        self.structure_list  = sorted(list(modality_present[2]))

        print(len(self.text_list),len(self.vision_list),len(self.structure_list))
        print(missing_rate,modal_ratio,num_nodes)

        mask = torch.zeros(N,3)
        mask[self.text_list,0] = 1
        mask[self.vision_list,1] = 1
        mask[self.structure_list,2] = 1
        return mask


    def split_modal_nodes_all(self,modal_ratio):
        N = self.text_feat.shape[0]
        num_modal = 3

        total_modal = N * num_modal
        num_keep = int(total_modal * modal_ratio)  # Total number of modality observations to retain.

        mask = torch.ones((N, num_modal), dtype=torch.int)

        # Remove complete modality columns first.
        for m in [0,1,2]: 

            mask[:, m] = 0
            # Stop once the remaining count reaches or falls below num_keep.

            if mask.sum() <= num_keep:
                break

        # Compute the number of observations to restore.
        remaining_keep = num_keep - mask.sum().item()

        if remaining_keep > 0:
            # Randomly restore remaining_keep observations in the current column.
            indices = np.random.choice(N, size=remaining_keep, replace=False)
            mask[indices, m] = 1

        self.text_list = torch.nonzero(mask[:,0])[:,0].numpy().tolist()
        self.vision_list = torch.nonzero(mask[:,1])[:,0].numpy().tolist()
        self.structure_list = torch.nonzero(mask[:,2])[:,0].numpy().tolist()


        return mask


    def split_modal_nodes_snowball2(self, modal_ratio, seed=0, max_hop=2, **kwargs):
        from utils import make_modality_mask
        return make_modality_mask(self.text_feat.shape[0], 3, 1-modal_ratio,
                                  'cluster', self.edge_index, seed, max_hop)


    def split_modal_nodes_missing(self, modal):


        N = self.text_feat.shape[0]
        if modal == 0:
            self.text_list, self.vision_list, self.structure_list =  [], [i for i in range(N)], [i for i in range(N)]
        if modal == 1:
            self.text_list, self.vision_list, self.structure_list =  [i for i in range(N)],[],  [i for i in range(N)]
        if modal == 2:
            self.text_list, self.vision_list, self.structure_list =  [i for i in range(N)],  [i for i in range(N)] ,[]

        mask = torch.zeros(N,3)
        mask[self.text_list,0] = 1
        mask[self.vision_list,1] = 1
        mask[self.structure_list,2] = 1
        return mask


    def split_modal_nodes(self, modal_ratio, seed=0):
        from utils import make_modality_mask
        mask = make_modality_mask(self.text_feat.shape[0], 3, 1-modal_ratio, seed=seed)
        self.text_list, self.vision_list, self.structure_list = [
            torch.where(mask[:, m] > 0)[0].tolist() for m in range(3)]
        return mask


    def identify(self, task, train_ratio):
        if not getattr(self, 'cached_split', False):
            raise ValueError('Prepare and load a saved supervision split before adaptation')
        if task != self.task or abs(train_ratio-self.split_ratio) > 1e-10:
            raise ValueError('Requested task/ratio differs from the cached split')


class Graph:
    def __init__(self,x,edge_index,is_tensor = False,labels = None):
        super(Graph, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.is_tensor = is_tensor
        self.labels = labels


    def to(self,device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.is_tensor = True


    def to_tensor(self,device):

        if self.is_tensor:return


        edge_index = torch.from_numpy(self.edge_index).to(device)

        x = torch.from_numpy(self.x).to(torch.float).to(device)

        self.x = x
        self.edge_index = edge_index
        self.is_tensor = True


        if not self.labels is None:
            if len(self.labels.shape) == 2:self.labels = torch.FloatTensor(self.labels).to(device)
            else: self.labels = torch.LongTensor(self.labels).to(device)


    def detach(self):
        return Graph(self.x.detach(),self.edge_index,True)


    def transformation_mlp(self,MLP):

        x = MLP(self.x)
        graph = Graph(x,self.edge_index,True)
        graph.original_x = self.x
        graph.adj = self.adj
        return graph


    def promptamation(self,modality_projections,train_dataset,name):

        xs = []
        for data in train_dataset:
            if data == name:xs.append(self.x)
            else: xs.append(modality_projections[data](self.x))
        x = torch.cat(xs,dim = 1)

        return Graph(x,self.edge_index,True)


    def node_split(self,label_num,instance_num):
        '''
        permutation = np.random.permutation(self.labels.shape[1])
        self.indices = np.random.choice(self.labels.shape[0],label_num,replace = False)
        self.indices.sort()
        labels = self.labels[:,permutation]
        self.train_nodes = np.unique(labels[self.indices,:instance_num].flatten())
        self.test_nodes = np.setdiff1d(np.unique(labels[self.indices].flatten()),self.train_nodes)
        '''
        self.indices = np.random.choice(list(self.labels.keys()),label_num,replace = False)
        self.indices.sort()
        self.train_nodes =[]
        for i in self.indices:self.train_nodes.extend(np.random.choice(np.setdiff1d(self.labels[i],self.indices),instance_num,replace=False))
        self.train_nodes = np.unique(self.train_nodes)
        self.test_nodes = []
        for i in self.indices:self.test_nodes.extend(self.labels[i])


        self.test_nodes = np.setdiff1d(np.unique(self.test_nodes),self.train_nodes)
        self.test_nodes = np.setdiff1d(self.test_nodes,self.indices)


        train_pos_link = []
        train_neg_link = []
        test_pos_link = []
        test_neg_link = []
        train_nodes = self.train_nodes
        test_nodes = self.test_nodes
        for i in train_nodes:
            for j in self.indices:
                if i in self.labels[j]:train_pos_link.append((i,j))
                else:train_neg_link.append((i,j))


        ground_truth = []
        for i in test_nodes:
            ground_truth.append([])
            for j in self.indices:
                if i in self.labels[j]:
                    test_pos_link.append((i,j))
                    ground_truth[-1].append(1)
                else:
                    test_neg_link.append((i,j))
                    ground_truth[-1].append(0)

        self.ground_truth = np.array(ground_truth)


        self.train_pos_edge_index = torch.tensor(np.array(train_pos_link).T).to(self.edge_index)
        self.train_neg_edge_index = torch.tensor(np.array(train_neg_link).T).to(self.edge_index)
        self.test_pos_edge_index = torch.tensor(np.array(test_pos_link).T).to(self.edge_index)
        self.test_neg_edge_index = torch.tensor(np.array(test_neg_link).T).to(self.edge_index)


    def attr_split(self,ratio):
        mask_node = np.random.choice(self.x.shape[0],int(self.x.shape[0]*ratio),replace=False)
        self.attr = self.x[mask_node,:].detach()
        self.x[mask_node] = 0
        self.mask = mask_node


    def attr_split2(self,ratio):
        mask_node = np.random.choice(self.x.shape[0],int(self.x.shape[0]*ratio),replace=False)

        self.xx = self.x.clone().detach()
        self.xx[mask_node] = 0
        self.mask = mask_node


    def link_split(self,train_ratio,test_ratio):

        row, col = self.edge_index


    # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(train_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        train_pos_edge_index = torch.stack([r, c], dim=0)
        train_pos_edge_index = to_undirected(train_pos_edge_index)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        val_pos_edge_index = torch.stack([r, c], dim=0)


        train_pos_edge_index = train_pos_edge_index.detach()
        val_pos_edge_index = val_pos_edge_index.detach()
        test_pos_edge_index = test_pos_edge_index.detach()

        num_nodes = self.x.shape[0]
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)

        self.train_edge_index = train_pos_edge_index
        self.test_edge_index = test_pos_edge_index


    def split_nodes(self,train_ratio,test_ratio):

        #### node split
        n_v = int(math.floor(train_ratio * self.x.size(0)))
        n_t = int(math.floor(test_ratio * self.x.size(0)))
        node_perm = torch.randperm(self.x.size(0))
        self.train_nodes = node_perm[:n_v]
        self.test_nodes = node_perm[n_v:n_v + n_t]


    def identify(self,task,train_ratio):


        if task == 'link':
            if self.is_tensor:
                edge_index = self.edge_index.detach()

            row = edge_index[0]
            col = edge_index[1]

            mask = row < col
            row, col = row[mask], col[mask]

            tmp = int(1/train_ratio)
            train_set = []
            tmp_set = []
            val_set = []
            test_set = []

            for i in range(row.size(0)):
                if i%tmp == 0:train_set.append(i)
                else:tmp_set.append(i)

            tmp = int((len(tmp_set)+0.0)/int(0.1*row.size(0)))
            for i,ele in enumerate(tmp_set):
                if i%tmp == 0:val_set.append(ele)
                else:test_set.append(ele)


            r, c = row[train_set], col[train_set]
            train_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[val_set], col[val_set]
            val_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[test_set], col[test_set]
            test_pos_edge_index = torch.stack([r, c], dim=0)

            train_pos_edge_index = train_pos_edge_index.detach()
            test_pos_edge_index = test_pos_edge_index.detach()
            val_pos_edge_index = val_pos_edge_index.detach()

            self.train_edge_index = train_pos_edge_index
            self.test_edge_index = test_pos_edge_index
            self.val_edge_index = val_pos_edge_index
            self.test_edge_index_negative = negative_sampling_identify(self.test_edge_index,self.x.size(0))

        else:


            train_num = int(self.labels.size(0)*train_ratio)
            val_num = int(self.labels.size(0)*0.1)

            if len(self.labels.shape) == 2:
                num_classes = int(self.labels.shape[1])

                label_arr = defaultdict(list)
                for index in range(num_classes):
                    label_arr[index] = np.where(self.labels[:,index].detach().cpu().numpy())[0].tolist()
                labels = []
                while len(label_arr)!=0:
                    for key in sorted(label_arr.keys()):
                        if len(label_arr[key]) == 0:
                            label_arr.pop(key)
                            continue
                        labels.append(label_arr[key].pop())


                train_set = set()
                while len(train_set)!=train_num:train_set.add(labels.pop(0))
                val_set = set()
                while len(val_set)!=val_num:
                    value = labels.pop(0)
                    if value not in train_set:val_set.add(value)
                test_set = []
                for i in range(self.labels.size(0)):
                    if i not in train_set and i not in val_set:test_set.append(i)

                train_set = list(train_set)
                val_set = list(val_set)


            else:
                num_classes = int(self.labels.max().item()+1)

                label_arr = defaultdict(list)
                for index,label in enumerate(self.labels):
                    label_arr[label.item()].append(index)


                labels = []
                while len(label_arr)!=0:
                    for key in sorted(label_arr.keys()):
                        if len(label_arr[key]) == 0:
                            label_arr.pop(key)
                            continue
                        labels.append(label_arr[key].pop())


                train_set = labels[:train_num]
                val_set = labels[train_num:train_num+val_num]
                test_set = labels[train_num+val_num:]


            self.train_labels=train_set
            self.test_labels = test_set
            self.val_labels = val_set


            print('train_labels:',len(self.train_labels),'  val_labels:',len(self.val_labels),' test_labels:',len(self.test_labels))


class dataGraph:

    def __init__(self,args):
        super(dataGraph, self).__init__()
        self.args = args
        self.dataset = args.dataset
        self.text = args.modal_text
        self.vision = args.modal_vision
        self.structure = args.modal_structure

        self.load_datasets()


    def load_datasets(self):
        root = Path(self.args.data_root) / self.dataset
        stem = split_stem(self.args.round_index, self.args.train_ratio, self.args.seed)
        with np.load(root/'splits'/(stem+'.npz'), allow_pickle=False) as saved:
            split = {key: saved[key].copy() for key in saved.files}
        self.adj_full = sp.load_npz(root/'adj_full.npz').tocsr()
        self.adj_full = self.adj_full.maximum(self.adj_full.T)
        self.adj_full.setdiag(0); self.adj_full.eliminate_zeros()
        excluded = split['sample']
        keep = np.ones(self.adj_full.shape[0], dtype=bool)
        keep[excluded] = False
        self.allowed_nodes = np.flatnonzero(keep)
        # Exclude the evaluation nodes and their incident edges.
        coo = self.adj_full.tocoo()
        valid = keep[coo.row] & keep[coo.col]
        self.adj_full = sp.csr_matrix((coo.data[valid], (coo.row[valid], coo.col[valid])),
                                     shape=coo.shape)
        self.original_feats = {
            'text': np.load(root/f'{self.text}_text_feature_raw.npy', mmap_mode='r'),
            'vision': np.load(root/f'{self.vision}_vision_feature_raw.npy', mmap_mode='r'),
            'structure': load_structure_features(root, self.structure,
                             str(split['task'].item()), stem, split)}
        for modal, feat in self.original_feats.items():
            if feat.ndim != 2 or feat.shape[0] != self.adj_full.shape[0]:
                raise ValueError(f'{modal} features must align with full-graph node ids')
        self.modals = ('text','vision','structure')
        self.rng = np.random.default_rng(self.args.seed + self.args.round_index)
        self.excluded_nodes = excluded
        self.split_digest = str(split['split_digest'].item())


    def fetch_for_fused(self,norm = True):

        subgraphs = {}

        for modal in self.modals:
            sampling_graph,_ = self.sampling(modal,self.args.fuse_scale,norm)
            sampling_graph.adj = _
            subgraphs[modal] = sampling_graph
        return subgraphs


    def fetch_mmg(self, norm=False):
        if norm:
            raise ValueError('Full-feature normalization is disabled; see README')
        edges, sample = self.sampling(self.args.subgraph_scale)
        graph = test_graph(self.original_feats['text'][sample],
                           self.original_feats['vision'][sample],
                           self.original_feats['structure'][sample], edges)
        graph.sample = sample
        return graph


    def fetch_subgraph(self,norm = True):
        subgraphs = {}

        for modal in self.modals:
            sampling_graph,_ = self.sampling(modal,self.args.subgraph_scale,norm)
            sampling_graph.adj = _
            subgraphs[modal] = sampling_graph
        return subgraphs


    def subgraph_to_tensor(self, subgraphs, device):
        for key in subgraphs.keys():
            subgraphs[key].to_tensor(device)


    def sampling(self, n_samples=512, norm=False):
        sample = sample_graph_nodes(self.adj_full, n_samples, self.rng, self.allowed_nodes)
        adj = self.adj_full[sample, :][:, sample].tocoo()
        edges = np.vstack((adj.row, adj.col)).astype(np.int64)
        return edges, sample


# Dataset preparation and split utilities.
DATASET_TASKS = {'books-lp':'link', 'books-nc':'node', 'cloth-copurchase':'link',
                 'sports-copurchase':'link', 'ele-fashion':'node', 'reddit-m':'node',
                 'arxiv':'node'}


def split_stem(index, train_ratio, seed=0):
    return f'test_graph{index}_train{train_ratio:g}_seed{seed}'


def canonical_edges(edge_index, num_nodes):
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('edge_index must have shape (2,E)')
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError('Edge endpoint outside node range')
    u, v = np.minimum(edges[0], edges[1]), np.maximum(edges[0], edges[1])
    code = np.unique(u[u != v]*num_nodes+v[u != v])
    return np.stack((code//num_nodes, code % num_nodes))


def undirected_edges(edges):
    return np.concatenate((edges, edges[::-1]), axis=1)


def array_digest(*arrays):
    import hashlib
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def sample_graph_nodes(adj, count, rng, allowed=None):
    """Random-walk induced sampling with restarts; handles disconnected graphs."""
    allowed = np.arange(adj.shape[0]) if allowed is None else np.asarray(allowed)
    if count < 1 or count > len(allowed):
        raise ValueError(f'Cannot sample {count} distinct nodes from {len(allowed)}')
    allowed_mask = np.zeros(adj.shape[0], dtype=bool)
    allowed_mask[allowed] = True
    visited = set()
    current = int(rng.choice(allowed))
    for step in range(max(100*count, 1000)):
        visited.add(current)
        if len(visited) == count:
            break
        neighbors = adj.indices[adj.indptr[current]:adj.indptr[current+1]]
        neighbors = neighbors[allowed_mask[neighbors]]
        if not len(neighbors) or rng.random() < 0.05:
            current = int(rng.choice(allowed))
        else:
            current = int(rng.choice(neighbors))
    if len(visited) < count:
        # Complete the node budget with additional restart seeds.
        rest = np.setdiff1d(allowed, np.fromiter(visited, dtype=np.int64))
        visited.update(rng.choice(rest, count-len(visited), replace=False).tolist())
    return np.array(sorted(visited), dtype=np.int64)


def sample_non_edges(full_edges, n, count, rng):
    """Sample unique undirected non-edges, excluding self-loops and known positive edges."""
    edges = canonical_edges(full_edges, n)
    positive = set((edges[0]*n+edges[1]).tolist())
    if count > n*(n-1)//2-len(positive):
        raise ValueError('Not enough distinct non-edges for this split')
    chosen, seen = [], set()
    for _ in range(10000):
        if len(chosen) == count:
            break
        candidates = rng.integers(n, size=(2, max(32, 2*(count-len(chosen)))))
        for u, v in candidates.T:
            if u == v:
                continue
            u, v = min(int(u), int(v)), max(int(u), int(v))
            code = u*n+v
            if code not in positive and code not in seen:
                chosen.append(code); seen.add(code)
                if len(chosen) == count:
                    break
    if len(chosen) != count:
        raise RuntimeError('Negative sampling exhausted; graph is too dense')
    code = np.asarray(chosen, dtype=np.int64)
    return np.stack((code//n, code % n))


def make_supervision_split(sample, edges, labels, task, ratio, rng):
    n = len(sample)
    if not 0 < ratio < 0.9:
        raise ValueError('train_ratio must be in (0,0.9)')
    full = canonical_edges(edges, n)
    result = {'sample': sample, 'task': np.array(task),
              'full_edge_index': undirected_edges(full)}
    if task == 'link':
        order = rng.permutation(full.shape[1])
        nt, nv = int(len(order)*ratio), int(len(order)*0.1)
        if min(nt, nv, len(order)-nt-nv) < 1:
            raise ValueError('Not enough edges for nonempty train/val/test splits')
        positive = [full[:, order[:nt]], full[:, order[nt:nt+nv]], full[:, order[nt+nv:]]]
        negative = sample_non_edges(full, n, len(order), rng)
        negatives = [negative[:, :nt], negative[:, nt:nt+nv], negative[:, nt+nv:]]
        for tag, pos, neg in zip(('train','val','test'), positive, negatives):
            result[tag+'_edge_index'] = pos
            result[tag+'_edge_index_negative'] = neg
        result['message_edge_index'] = undirected_edges(positive[0])
    else:
        if labels is None or len(labels) != n:
            raise ValueError('Node classification needs aligned labels')
        if labels.ndim == 1:
            # Seeded, class-interleaved node splitting.
            groups = [rng.permutation(np.flatnonzero(labels == c)).tolist()
                      for c in np.unique(labels[labels >= 0])]
            order = []
            while any(groups):
                for group in groups:
                    if group:
                        order.append(group.pop())
            order = np.asarray(order, dtype=np.int64)
        else:
            order = rng.permutation(n)
        nt, nv = int(len(order)*ratio), int(len(order)*0.1)
        if min(nt, nv, len(order)-nt-nv) < 1:
            raise ValueError('Not enough labeled nodes for train/val/test')
        for tag, index in zip(('train','val','test'), (order[:nt], order[nt:nt+nv], order[nt+nv:])):
            result[tag+'_labels'] = index
        result['message_edge_index'] = undirected_edges(full)
    return result


def load_structure_features(root, structure, task, stem, split):
    import json
    if task == 'link':
        path = root/'structure'/f'{structure}_{stem}.npy'
        metadata = path.with_suffix('.json')
        if not path.exists() or not metadata.exists():
            raise FileNotFoundError(
                f'{path}: prepare split-specific Node2Vec offline using node2vec.py; '
                'a full-graph embedding can contain validation/test edges')
        with metadata.open() as f:
            info = json.load(f)
        if info.get('reference_digest') != str(split['reference_digest'].item()):
            raise ValueError('Node2Vec provenance does not match the cached reference graph')
    else:
        path = root/f'{structure}_structure_feature_raw.npy'
    return np.load(path, mmap_mode='r')


def load_cached_mask(data_root, dataset, stem, mode, missing_rate, num_nodes):
    path = Path(data_root)/dataset/'masks'/f'{stem}_{mode}_mr{missing_rate:g}.npy'
    mask = np.load(path, allow_pickle=False)
    if mask.shape != (num_nodes, 3) or not np.isin(mask, (0,1)).all() or (mask.sum(1)<1).any():
        raise ValueError(f'Invalid observation mask: {path}')
    expected = max(num_nodes, int(round(num_nodes*3*(1-missing_rate))))
    if int(mask.sum()) != expected:
        raise ValueError(f'Mask observation budget differs from the requested rate: {path}')
    return torch.from_numpy(mask.astype(np.float32))


def prepare_dataset_cache(data_root, dataset, rounds=range(10), eval_nodes=5000,
                          ratios=(0.01,0.02), missing_rates=(0.2,0.4,0.6), seed=0):
    """Prepare sampled graphs, supervision splits, modality masks and reference graphs."""
    from utils import make_modality_mask
    root = Path(data_root)/dataset
    task = DATASET_TASKS[dataset]
    adj = sp.load_npz(root/'adj_full.npz').tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0); adj.eliminate_zeros()
    labels = None
    if task == 'node':
        if (root/'labels.npy').exists():
            labels = np.load(root/'labels.npy')
        elif (root/'labels-w-missing.pt').exists():
            labels = torch.load(root/'labels-w-missing.pt', map_location='cpu', weights_only=True)
            labels = np.asarray(labels)
    for directory in ('test_graphs','splits','masks','structure'):
        (root/directory).mkdir(parents=True, exist_ok=True)
    for index in rounds:
        path = root/'test_graphs'/f'test_graph{index}.npz'
        if path.exists():
            with np.load(path, allow_pickle=True) as saved:
                if 'sample' not in saved:
                    raise ValueError(f'{path} must contain original node ids in sample')
                sample = np.asarray(saved['sample'], dtype=np.int64)
                node_labels = saved['labels'] if 'labels' in saved else None
                if node_labels is not None and node_labels.size == 0:
                    node_labels = None
        else:
            sample = sample_graph_nodes(adj, eval_nodes, np.random.default_rng(seed+index))
            node_labels = None if labels is None else labels[sample]
        if len(sample) != eval_nodes or len(np.unique(sample)) != len(sample):
            raise ValueError(f'{path}: expected {eval_nodes} distinct sampled node ids')
        if sample.min() < 0 or sample.max() >= adj.shape[0]:
            raise ValueError(f'{path}: invalid original node ids')
        local = adj[sample, :][:, sample].tocoo()
        edges = np.vstack((local.row, local.col)).astype(np.int64)
        if node_labels is None and labels is not None:
            node_labels = labels[sample]
        if not path.exists():
            np.savez_compressed(path, sample=sample, edge_index=edges,
                                labels=np.array([]) if node_labels is None else node_labels)
        for ratio in ratios:
            stem = split_stem(index, ratio, seed)
            split_path = root/'splits'/(stem+'.npz')
            if split_path.exists():
                with np.load(split_path, allow_pickle=False) as saved:
                    split = {k: saved[k].copy() for k in saved.files}
                if not np.array_equal(sample, split['sample']):
                    raise ValueError('Saved supervision split refers to different nodes')
            else:
                split = make_supervision_split(sample, edges, node_labels, task, ratio,
                                               np.random.default_rng(seed+1009*index))
                if task == 'link':
                    removed = np.concatenate((split['val_edge_index'], split['test_edge_index']), axis=1)
                    global_removed = sample[removed]
                    coo = adj.tocoo()
                    n = adj.shape[0]
                    removal_codes = np.concatenate((global_removed[0]*n+global_removed[1],
                                                    global_removed[1]*n+global_removed[0]))
                    keep = ~np.isin(coo.row*n+coo.col, removal_codes)
                    reference = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape)
                    reference.sort_indices()
                    sp.save_npz(root/'splits'/(stem+'_reference.npz'), reference)
                    split['reference_digest'] = np.array(array_digest(reference.indptr, reference.indices))
                else:
                    split['reference_digest'] = np.array('node-classification-full-reference')
                split['split_digest'] = np.array(array_digest(*[split[k] for k in sorted(split)]))
                np.savez_compressed(split_path, **split)
            for mode in ('random','degree','cluster'):
                for rate in missing_rates:
                    mask_path = root/'masks'/f'{stem}_{mode}_mr{rate:g}.npy'
                    if not mask_path.exists():
                        mask = make_modality_mask(len(sample), 3, rate, mode,
                                                  split['message_edge_index'], seed+1009*index)
                        np.save(mask_path, mask.numpy())

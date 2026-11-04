import argparse
import json
from pathlib import Path
from data.load_data import DATASET_TASKS, split_stem, array_digest, prepare_dataset_cache

import torch
from torch_geometric.nn import Node2Vec
import numpy as np
import scipy.sparse as sp
import os
from random import randint
import random


def save_embedding(model, output_path, metadata=None):
    """Save node embeddings and reference-graph metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, model.embedding.weight.detach().cpu().numpy())
    if metadata is not None:
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Offline Node2Vec on cached reference graphs')
    parser.add_argument('--data_root', default='dataset')
    parser.add_argument('--dataset', choices=sorted(DATASET_TASKS), required=True)
    parser.add_argument('--rounds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--train_ratios', nargs='+', type=float, default=[0.01,0.02])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cpu' if args.device < 0 else f'cuda:{args.device}')
    root = Path(args.data_root)/args.dataset
    jobs = []
    if DATASET_TASKS[args.dataset] == 'link':
        for index in args.rounds:
            for ratio in args.train_ratios:
                stem = split_stem(index, ratio, args.seed)
                with np.load(root/'splits'/(stem+'.npz'), allow_pickle=False) as saved:
                    digest = str(saved['reference_digest'].item())
                jobs.append((root/'splits'/(stem+'_reference.npz'),
                             root/'structure'/f'node2vec_{stem}.npy', digest))
    else:
        jobs.append((root/'adj_full.npz', root/'node2vec_structure_feature_raw.npy', None))
    for source, destination, expected_digest in jobs:
        if destination.exists():
            # Reuse cached embeddings.
            if expected_digest is not None:
                metadata = json.loads(destination.with_suffix('.json').read_text())
                if metadata['reference_digest'] != expected_digest:
                    raise ValueError('Existing Node2Vec artifact belongs to another reference graph')
            print(f'Reuse {destination}')
            continue
        adj = sp.load_npz(source).tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0); adj.eliminate_zeros(); adj.sort_indices()
        digest = array_digest(adj.indptr, adj.indices)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError('Reference graph differs from the cached supervision split')
        coo = adj.tocoo()
        edge_index = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        # Use the full node count to preserve feature-row alignment.

        model = Node2Vec(edge_index, args.embedding_dim, args.walk_length,
                         args.context_size, args.walks_per_node, sparse=True,
                         num_nodes=adj.shape[0]).to(device)
        loader = model.loader(batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
        model.train()
        for epoch in range(args.epochs):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward(); optimizer.step()
            print(f'{destination.name}: epoch {epoch+1}/{args.epochs}')
        save_embedding(model, destination, {'reference_digest': digest,
                       'reference_graph': str(source), 'num_nodes': adj.shape[0],
                       'embedding_dim': args.embedding_dim, 'seed': args.seed})


def get_test_graph(dataset, data_root='dataset', seed=0):

    prepare_dataset_cache(data_root, dataset, seed=seed)


def change_pt_to_npy(dataset, data_root='dataset'):
    """Export node-classification embeddings in NumPy format."""
    if DATASET_TASKS[dataset] == 'link':
        raise ValueError('Link prediction requires split-specific Node2Vec artifacts')
    root = Path(data_root)/dataset
    value = torch.load(root/'embedding.pt', map_location='cpu', weights_only=True)
    np.save(root/'node2vec_structure_feature_raw.npy', value.detach().cpu().numpy())


if __name__ == '__main__':
    main()

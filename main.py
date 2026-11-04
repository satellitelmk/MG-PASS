"""Command-line entry points for MG-PASS pre-training and downstream tasks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from data.load_data import (DATASET_TASKS, dataGraph, load_test_graph,
                            load_cached_mask, split_stem)
from models.models import (MLP, TwoLayerGraphEncoder, GraphConditionedSemanticSpace,
                           SemanticFusedGNN, SemanticCalibrationAdapter)
from models.autoencoder import MyGAE, MyTask, MLPDecoder2, InnerProduct3Decoder
from train import pretrain_epoch, calibrate_semantic_space, prepare_pretraining_targets, evaluate_graph
from utils import seed_everything

ARCHITECTURE_KEYS = ('share_dims', 'hidden_dims', 'output_dims', 'semantic_space_size',
                     'num_layers', 'top_k', 'tau', 'gamma', 'eta')


def build_parser():
    parser = argparse.ArgumentParser(description='MG-PASS')
    parser.add_argument('--mode', choices=['pretrain','adapt','evaluate','all'], default='pretrain')
    parser.add_argument('--data_root', default='dataset')
    parser.add_argument('--output_root', default='outputs')
    parser.add_argument('--dataset', choices=sorted(DATASET_TASKS), default='cloth-copurchase')
    parser.add_argument('--rounds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--modal_text', default='clip')
    parser.add_argument('--modal_vision', default='clip')
    parser.add_argument('--modal_structure', default='node2vec')
    parser.add_argument('--pretrain_task', choices=['EdgePred','AttrMask'], default='EdgePred',
                        help='Self-supervised pre-training objective')
    parser.add_argument('--share_dims', type=int, default=256)
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--output_dims', type=int, default=256)
    parser.add_argument('--semantic_space_size', type=int, default=1024,
                        help='Number of basic semantic factors T, Eq. (5)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5, help='Semantic fusion weight, Eq. 13')
    parser.add_argument('--eta', type=float, default=0.25)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--model_lr', type=float, default=0.001)
    parser.add_argument('--constructor_lr', type=float, default=0.01,
                        help='Learning rate for the challenging view constructor')
    parser.add_argument('--constructor_ascent_steps', type=int, default=5,
                        help='Number of constructor gradient-ascent steps J, Eq. (4)')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--subgraph_scale', type=int, default=512)
    parser.add_argument('--subgraphs_per_epoch', type=int, default=1)
    parser.add_argument('--edge_mask_rate', type=float, default=0.2)
    parser.add_argument('--random_missing_min', type=float, default=0.0)
    parser.add_argument('--random_missing_max', type=float, default=2/3)
    parser.add_argument('--adapter_rank', type=int, default=64,
                        help='Bottleneck dimension r for semantic calibration, Eq. (18)')
    parser.add_argument('--adapt_lr', type=float, default=0.001)
    parser.add_argument('--adapt_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--train_ratio', type=float, default=0.01,
                        help='Downstream supervision ratio used to select the cached split')
    parser.add_argument('--missing_mode', choices=['random','degree','cluster'], default='random')
    parser.add_argument('--missing_rate', type=float, default=0.6)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=-1, help='-1 for CPU, otherwise CUDA device index')
    return parser


def _encoder(args):
    return SemanticFusedGNN(args.share_dims, args.output_dims,
        num_layers=args.num_layers, topk=args.top_k, tau=args.tau,
        gamma=args.gamma, eta=args.eta)


def _projections(args, dims, device):
    return nn.ModuleDict({modal: MLP(dim, args.hidden_dims, args.share_dims)
                          for modal, dim in zip(args.modal_names, dims)}).to(device)


def _run_dir(args, index):
    return Path(args.output_root)/args.dataset/split_stem(index, args.train_ratio, args.seed)


def _pretrain_path(args, index):
    return _run_dir(args, index)/f'pretrain_{args.pretrain_task}.pth'


def _adapt_path(args, index):
    return _run_dir(args, index)/(f'adapt_{args.pretrain_task}_{args.missing_mode}'
                                  f'_mr{args.missing_rate:g}.pth')


def pretrain(args, device, index, norm=False):
    """Pre-train MG-PASS with challenging and random modality views."""
    args = deepcopy(args)
    args.round_index = index
    seed_everything(args.seed+index)
    dataloader = dataGraph(args)
    dims = [int(dataloader.original_feats[m].shape[1]) for m in args.modal_names]
    model = MyGAE(_encoder(args)).to(device)
    modality_projections = _projections(args, dims, device)
    semantic_space = GraphConditionedSemanticSpace(args, sum(dims)).to(device)
    view_constructor = TwoLayerGraphEncoder(args, sum(dims), len(args.modal_names), 'GCN').to(device)
    auxiliary = nn.ModuleDict({m: nn.Linear(args.output_dims, d)
                               for m, d in zip(args.modal_names, dims)}).to(device)
    if args.pretrain_task == 'EdgePred':
        auxiliary = nn.ModuleDict().to(device)
    params = list(model.parameters())+list(semantic_space.parameters())+list(modality_projections.parameters())+list(auxiliary.parameters())
    optimizer = torch.optim.Adam(params, lr=args.model_lr)
    constructor_optimizer = torch.optim.Adam(view_constructor.parameters(), lr=args.constructor_lr)
    directory = _run_dir(args, index)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/f'pretrain_{args.pretrain_task}.csv').open('w') as log:
        log.write('epoch,challenging_loss,random_loss,grounding_loss,total_loss\n')
        for epoch in range(1, args.epochs+1):
            subgraphs = []
            for _ in range(args.subgraphs_per_epoch):
                graph = dataloader.fetch_mmg(norm=norm).to_tensor(device)
                prepare_pretraining_targets(graph, args.pretrain_task, args.edge_mask_rate)
                subgraphs.append(graph)
            _, _, loss = pretrain_epoch(args, model, semantic_space, view_constructor, modality_projections, auxiliary,
                                      subgraphs, optimizer, epoch, device, log, constructor_optimizer)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:

                state = {m: modality_projections[m].state_dict() for m in args.modal_names}
                state.update(format_version=3, checkpoint_id=str(uuid4()), model=model.encoder.state_dict(),
                    semantic_space=semantic_space.state_dict(), view_constructor=view_constructor.state_dict(),
                    auxiliary=auxiliary.state_dict(), optimizer=optimizer.state_dict(),
                    constructor_optimizer=constructor_optimizer.state_dict(), config=vars(args),
                    dims=dims, epoch=epoch, loss=loss, split_digest=dataloader.split_digest)
                torch.save(state, _pretrain_path(args, index))
    return str(_pretrain_path(args, index))


def _load_adaptation(args, device, index, ratio, modal_ratio):
    seed_everything(args.seed+index)
    graph = load_test_graph(args.dataset, index, args.modal_text, args.modal_vision,
            args.modal_structure, data_root=args.data_root, train_ratio=ratio, seed=args.seed)
    graph.to_tensor(device)
    graph.mask = load_cached_mask(args.data_root, args.dataset, graph.split_stem,
                args.missing_mode, 1-modal_ratio, graph.text_feat.size(0)).to(device)
    state = torch.load(_pretrain_path(args, index), map_location=device, weights_only=True)
    if state.get('format_version') != 3:
        raise ValueError('Unsupported checkpoint format_version; expected 3')
    if state['split_digest'] != graph.split_digest:
        raise ValueError('Pretrained checkpoint and cached evaluation split do not match')
    graph.pretrain_id = state['checkpoint_id']
    model_args = deepcopy(args)
    for key in ARCHITECTURE_KEYS:
        setattr(model_args, key, state['config'][key])
    for key in ('dataset','modal_text','modal_vision','modal_structure','pretrain_task'):
        if getattr(args, key) != state['config'][key]:
            raise ValueError(f'Checkpoint configuration differs: {key}')
    dims = [graph.text_feat.size(1), graph.vision_feat.size(1), graph.structure_feat.size(1)]
    if dims != state['dims']:
        raise ValueError('Feature dimensions differ from pretraining')
    modality_projections = _projections(model_args, dims, device)
    for modal in args.modal_names:
        modality_projections[modal].load_state_dict(state[modal])
    encoder = _encoder(model_args)
    semantic_space = GraphConditionedSemanticSpace(model_args, sum(dims)).to(device)
    semantic_space.load_state_dict(state['semantic_space'])
    dim = model_args.output_dims*len(args.modal_names)
    if graph.task == 'link':
        model = MyTask('link', encoder, decoder=InnerProduct3Decoder(dim, model_args.output_dims)).to(device)
    else:
        if graph.labels is None:
            raise ValueError('Node classification requires labels')
        multi = graph.labels.ndim == 2
        classes = graph.labels.size(1) if multi else int(graph.labels.max().item()+1)
        model = MyTask('node_m' if multi else 'node', encoder,
                      decoder=MLPDecoder2(dim, classes)).to(device)
    # Load encoder weights after task-model initialization.
    model.encoder.load_state_dict(state['model'])
    calibration_adapter = SemanticCalibrationAdapter(model_args.share_dims, args.adapter_rank).to(device)
    for module in [model.encoder, semantic_space, modality_projections]:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return graph, modality_projections, model, semantic_space, calibration_adapter


def adapt(args, device, index, ratio=0.01, modal_ratio=0.4, num=None,
                 norm=False, tune=False):
    """Optimize the semantic calibration adapter and task prediction head."""
    if tune or norm:
        raise ValueError('Adaptation requires tune=False and norm=False')
    if num is not None and num != index:
        raise ValueError('Each round must use its own independently pretrained checkpoint')
    graph, modality_projections, model, semantic_space, calibration_adapter = _load_adaptation(args, device, index, ratio, modal_ratio)
    optimizer = torch.optim.Adam(list(calibration_adapter.parameters())+list(model.decoder.parameters()),
                                 lr=args.adapt_lr)
    with _adapt_path(args, index).with_suffix('.csv').open('w') as log:
        log.write('epoch,loss,validation_score\n')
        result = calibrate_semantic_space(modality_projections, semantic_space, model, graph, optimizer, device, log,
                     calibration_adapter=calibration_adapter, epochs=args.adapt_epochs, patience=args.patience,
                     checkpoint_path=_adapt_path(args, index))
    result.update(round=index, train_ratio=ratio, missing_mode=args.missing_mode,
                  missing_rate=1-modal_ratio, split_digest=graph.split_digest)
    _adapt_path(args, index).with_suffix('.json').write_text(json.dumps(result, indent=2))
    return result


def adapt_with_missingness(args, device, index, ratio=0.01, modal_ratio=0.4, num=None,
                  mode='random', norm=False, tune=False):

    args = deepcopy(args)
    args.missing_mode = {'snowball':'cluster'}.get(mode, mode)
    return adapt(args, device, index, ratio, modal_ratio, num, norm, tune)


def evaluate_saved(args, device, index):
    graph, modality_projections, model, semantic_space, calibration_adapter = _load_adaptation(
        args, device, index, args.train_ratio, 1-args.missing_rate)
    saved = torch.load(_adapt_path(args, index), map_location=device, weights_only=True)
    if saved['split_digest'] != graph.split_digest:
        raise ValueError('Adaptation checkpoint uses another split')
    if saved['pretrain_id'] != graph.pretrain_id:
        raise ValueError('Pretrained weights changed after this adaptation checkpoint was saved')
    calibration_adapter.load_state_dict(saved['calibration_adapter'])
    model.decoder.load_state_dict(saved['decoder'])
    return {'round': index, 'test_score': evaluate_graph(modality_projections, semantic_space, model, graph, calibration_adapter)}


def main():
    args = build_parser().parse_args()
    args.modal_names = ['text','vision','structure']
    args.dataset_task = DATASET_TASKS
    if args.share_dims != args.output_dims:
        raise ValueError('share_dims must equal output_dims for Eq. (8) residual evolution')
    if not (0 <= args.random_missing_min <= args.random_missing_max <= 2/3):
        raise ValueError('Random missingness interval must be inside [0,2/3]')
    if not 0 <= args.gamma <= 1 or not 1 <= args.top_k <= args.semantic_space_size:
        raise ValueError('Invalid fusion weight or top_k')
    if min(args.epochs, args.adapt_epochs, args.checkpoint_every, args.subgraphs_per_epoch,
           args.constructor_ascent_steps, args.num_layers) < 1:
        raise ValueError('Epochs, layers, checkpoint interval and ascent steps must be positive')
    if not 0 < args.edge_mask_rate < 1 or args.tau <= 0:
        raise ValueError('Invalid edge mask rate or temperature')
    if args.cuda >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --cuda -1 for CPU')
    device = torch.device('cpu' if args.cuda < 0 else f'cuda:{args.cuda}')
    results = []
    for index in args.rounds:
        if args.mode in ('pretrain','all'):
            pretrain(args, device, index)
        if args.mode in ('adapt','all'):
            results.append(adapt(args, device, index, args.train_ratio, 1-args.missing_rate))
        if args.mode == 'evaluate':
            results.append(evaluate_saved(args, device, index))
    if results:
        scores = [r['test_score'] for r in results]
        report = {'rounds': results, 'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
                  'score_scale':'0-1', 'num_rounds':len(scores)}
        destination = Path(args.output_root)/args.dataset
        destination.mkdir(parents=True, exist_ok=True)
        name = (f'{args.mode}_{args.pretrain_task}_train{args.train_ratio:g}_'
                f'{args.missing_mode}_mr{args.missing_rate:g}_seed{args.seed}.json')
        (destination/name).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

import math
import random
from urllib.parse import ParseResultBytes
from sklearn import multiclass

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from torch_scatter import scatter
from sklearn.metrics import f1_score,roc_auc_score

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        torch.nn.init.zeros_(tensor)


EPS = 1e-15
LOG_VAR_MAX = 10
LOG_VAR_MIN = EPS

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def negative_sampling(pos_edge_index, num_nodes):


    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1]) # Flatten edge coordinates.
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero(as_tuple = False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        perm[rest] = tmp
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero(as_tuple = False).view(-1)

    row, col = torch.div(perm, num_nodes,rounding_mode='floor'), perm % num_nodes   


    return torch.stack([row, col], dim=0).to(pos_edge_index.device)


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


class MLPDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MLPDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        glorot(self.weight)
        zeros(self.bias)

    def forward(self,x):
        return torch.matmul(x, self.weight)+self.bias


class MLPDecoder2(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MLPDecoder2, self).__init__()

        from models.models import MLP
        self.mlp = MLP(in_channels,in_channels,out_channels)
        self.mlp.reset_parameters()
        self.mlp.Dropout.p = 0.5


    def forward(self,x):
        return self.mlp(x)


class InnerProductTestDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self, in_channels, out_channels):
        super(InnerProductTestDecoder, self).__init__()


        self.mlp = nn.Linear(in_channels,out_channels)
        self.mlp.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.mlp(z)
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class InnerProduct3Decoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self, in_channels, out_channels):
        super(InnerProduct3Decoder, self).__init__()


        self.mlp = nn.Linear(in_channels,out_channels)
        self.mlp.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class InnerProduct2Decoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self, in_channels, out_channels):
        super(InnerProduct2Decoder, self).__init__()


        self.text_mlp = nn.Linear(in_channels,out_channels)
        self.vision_mlp = nn.Linear(in_channels,out_channels)
        self.structure_mlp = nn.Linear(in_channels,out_channels)

        self.text_mlp.reset_parameters()
        self.vision_mlp.reset_parameters()
        self.structure_mlp.reset_parameters()


    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """


        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class MyGAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(MyGAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder


        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.encoder, 'reset_parameters'):
            self.encoder.reset_parameters()
        if hasattr(self.decoder, 'reset_parameters'):
            self.decoder.reset_parameters()


    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)


    def naive_encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder.naive_forward(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple = False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None,
                   all_pos_edge_index=None):
        """Stable EdgePred BCE; negatives exclude every known positive edge.

        A supplied negative set lets both views and constructor steps share the
        same targets. all_pos_edge_index includes held-out reconstruction targets.
        """
        from torch_geometric.utils import negative_sampling as pyg_negative_sampling
        if pos_edge_index.numel() == 0:
            raise ValueError('EdgePred needs at least one positive target')
        if neg_edge_index is None:
            known = pos_edge_index if all_pos_edge_index is None else all_pos_edge_index
            known = to_undirected(known, num_nodes=z.size(0))
            neg_edge_index = pyg_negative_sampling(
                known, num_nodes=z.size(0), num_neg_samples=pos_edge_index.size(1),
                method='sparse')
        if neg_edge_index.numel() == 0:
            raise ValueError('No non-edges available for negative sampling')
        positive = self.decoder(z, pos_edge_index, sigmoid=False)
        negative = self.decoder(z, neg_edge_index, sigmoid=False)
        return F.softplus(-positive).mean() + F.softplus(negative).mean()


    def recon_loss_isolate(self, z_text,z_vision,z_structure, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """


        if 3*z_text.size(0) < pos_edge_index.size(1):
            pos_edge_index = pos_edge_index[:,np.random.choice(pos_edge_index.size(1),3*z_text.size(0),replace=False)]
        neg_edge_index = negative_sampling(pos_edge_index, z_text.size(0))


        pos_loss_text = -torch.log(self.decoder(z_text, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss_text = -torch.log(1 - self.decoder(z_text, neg_edge_index, sigmoid=True) + EPS).mean()

        pos_loss_vision= -torch.log(self.decoder(z_vision, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss_vision= -torch.log(1 - self.decoder(z_vision, neg_edge_index, sigmoid=True) + EPS).mean()

        pos_loss_structure = -torch.log(self.decoder(z_structure, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss_structure = -torch.log(1 - self.decoder(z_structure, neg_edge_index, sigmoid=True) + EPS).mean()

        return (pos_loss_text+ pos_loss_vision + pos_loss_structure + neg_loss_text+neg_loss_vision+neg_loss_structure)/6


    def recon_loss_isolate2(self, z_text,z_vision,z_structure, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """


        if 3*z_text.size(0) < pos_edge_index.size(1):
            pos_edge_index = pos_edge_index[:,np.random.choice(pos_edge_index.size(1),3*z_text.size(0),replace=False)]
        neg_edge_index = negative_sampling(pos_edge_index, z_text.size(0))

        z = (z_text+z_vision+z_structure)/3


        pos_loss= -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss= -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return (pos_loss+  neg_loss)


    def recon_loss2(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()


        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))


        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()


        return pos_loss + neg_loss

    def class_test(self,x,target_nodes,test_nodes,labels):


        sources = target_nodes * len(test_nodes)
        targets = []
        for i in test_nodes:
            targets.extend([i]*len(target_nodes))

        edge_index = torch.tensor(np.array([sources,targets]))
        outputs = self.decoder(x, edge_index, sigmoid=True).view(len(test_nodes),-1)

        if self.task == 'node_m':

            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
        else:
            outputs = outputs.argmax(dim=1)
        test_value = f1_score(labels.cpu().detach(), outputs.cpu().detach(), average='micro')
        return test_value
    def class_loss(self,z, pos_edge_index,neg_edge_index):

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return (pos_loss/3 + neg_loss)


    def class_loss2(self,z, pos_edge_index,neg_edge_index):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        tmp = z[2000:]
        reg = torch.norm(torch.mm(tmp,tmp.T)-torch.eye(tmp.shape[0]).to(z))
        return (pos_loss/3 + neg_loss) + 0.01*reg


    def class_loss3(self,z, pos_edge_index,neg_edge_index):

        output = z[:2000]@(z[2000:].t())
        return nn.CrossEntropyLoss()(output[pos_edge_index[0]],pos_edge_index[1]-2000)


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class MyTask(MyGAE):
    def __init__(self,task,encoder,share_dims=0, output_dims=0,decoder = None):
        if task == 'link':
            print(decoder)
            if decoder is not None:super(MyTask,self).__init__(encoder,decoder=decoder)
            else: super(MyTask,self).__init__(encoder,decoder=None)
        else:
            if decoder is not None:super(MyTask,self).__init__(encoder,decoder=decoder)
            else: super(MyTask,self).__init__(encoder=encoder,decoder=MLPDecoder(share_dims, output_dims))
            if task == 'node':self.loss = nn.CrossEntropyLoss()
            if task == 'node_m':self.loss = nn.BCEWithLogitsLoss()
        self.task = task

    def class_loss(self,x,target):

        outputs = self.decoder(x)
        return self.loss(outputs,target)

    def class_loss_isolate(self,x_text,x_vision,x_structure,target):

        outputs_text = self.decoder(x_text)
        outputs_vision = self.decoder(x_vision)
        outputs_structure = self.decoder(x_structure)
        return self.loss(outputs_text,target)+self.loss(outputs_vision,target)+self.loss(outputs_structure,target)


    def class_test(self,x,target):


        outputs = self.decoder(x)
        if self.task == 'node_m':

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)

        test_value = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')
        return test_value


    def class_test_all(self,x,target):


        outputs = self.decoder(x)


        if self.task == 'node':
            labels = target.cpu().detach().numpy()


            la = np.unique(labels)
            if len(la) != outputs.shape[1]:
                dic = OrderedDict(zip(la,np.arange(len(la))))
                labels = [dic[_] for _ in labels]
                auc= roc_auc_score(labels,torch.softmax(outputs[:,la],dim=1).cpu().detach(), average='macro',multi_class='ovr')
            else: auc= roc_auc_score(target.cpu().detach(),torch.softmax(outputs,dim=1).cpu().detach(), average='macro',multi_class='ovr')
        else: auc= roc_auc_score(target.cpu().detach(),torch.sigmoid(outputs).cpu().detach(), average='macro',multi_class='ovr')
        if self.task == 'node_m':

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)

        f1 = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')


        return (f1,auc)


    def test_reduce(self, z, test_map, test_neg_map):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_edge_index = test_map[1]
        value_pos = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_pred = scatter(value_pos,index=test_map[0], dim=0, reduce="mean")

        neg_edge_index = test_neg_map[1]
        value_neg = self.decoder(z, neg_edge_index, sigmoid=True)
        neg_pred = scatter(value_neg,index=test_neg_map[0], dim=0, reduce="mean")


        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = z.new_ones(pos_pred.size(0))
        neg_y = z.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


    def test_isolate(self, z_text,z_vision,z_structure, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """

        pos_pred_text = self.decoder(z_text, pos_edge_index, sigmoid=True)
        neg_pred_text = self.decoder(z_text, neg_edge_index, sigmoid=True)

        pos_pred_vision = self.decoder(z_vision, pos_edge_index, sigmoid=True)
        neg_pred_vision = self.decoder(z_vision, neg_edge_index, sigmoid=True)

        pos_pred_structure = self.decoder(z_structure, pos_edge_index, sigmoid=True)
        neg_pred_structure = self.decoder(z_structure, neg_edge_index, sigmoid=True)

        pos_pred = (pos_pred_text+pos_pred_vision+pos_pred_structure)/3
        neg_pred = (neg_pred_text+neg_pred_vision+neg_pred_structure)/3

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = z_text.new_ones(pos_pred.size(0))
        neg_y = z_text.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


    def test_isolate2(self, z_text,z_vision,z_structure, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """

        z = (z_text+z_vision+z_structure)/3
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = z_text.new_ones(pos_pred.size(0))
        neg_y = z_text.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


    def class_test_reduce(self,x,target,index):


        outputs = self.decoder(x)
        outputs = torch.softmax(outputs,dim=1)
        outputs = scatter(outputs,index= index, dim=0, reduce="mean")
        outputs = outputs.argmax(dim=1)

        test_value = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')
        return test_value


    def class_test_isolate(self,x_text,x_vision,x_structure,target):


        if self.task == 'node_m':
            outputs_text = self.decoder(x_text)
            outputs_vision = self.decoder(x_vision)
            outputs_structure = self.decoder(x_structure)

            outputs = (outputs_text+outputs_vision+outputs_structure)/3

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs_text = self.decoder(x_text)
            outputs_text = torch.softmax(outputs_text,dim=1)

            outputs_vision = self.decoder(x_vision)
            outputs_vision = torch.softmax(outputs_vision,dim=1)

            outputs_structure = self.decoder(x_structure)
            outputs_structure = torch.softmax(outputs_structure,dim=1)

            outputs = (outputs_text+outputs_vision+outputs_structure)/3
            outputs = outputs.argmax(dim=1)


        test_value = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')
        return test_value


    def class_test_isolate2(self,x_text,x_vision,x_structure,target):


        outputs_text = self.decoder(x_text)
        outputs_vision = self.decoder(x_vision)
        outputs_structure = self.decoder(x_structure)

        outputs = torch.softmax((outputs_text+outputs_vision+outputs_structure)/3,dim=1)
        outputs = outputs.argmax(dim=1)

        test_value = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')
        return test_value


    def GP_link_loss(self,graph,z,pos_edge_index):
        if 3*z.size(0) < pos_edge_index.size(1):
            pos_edge_index = pos_edge_index[:,np.random.choice(pos_edge_index.size(1),3*z.size(0),replace=False)]
        pos_loss = -torch.log(self.decoder(graph = graph,z = z, edge_index= pos_edge_index, sigmoid=True) + EPS).mean()
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - self.decoder(graph = graph,z = z, edge_index= neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss


    def GP_link_test(self,graph,z,pos_edge_index,neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(graph,z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(graph,z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def GP_node_loss(self,graph,x,target):
        outputs = self.decoder(graph,x)[graph.train_labels]
        return self.loss(outputs,target)

    def GP_node_test(self,graph,x,target):
        outputs = self.decoder(graph,x)[graph.test_labels]
        if self.task == 'node':
            labels = target.cpu().detach().numpy()


            la = np.unique(labels)
            if len(la) != outputs.shape[1]:
                dic = OrderedDict(zip(la,np.arange(len(la))))
                labels = [dic[_] for _ in labels]
                auc= roc_auc_score(labels,torch.softmax(outputs[:,la],dim=1).cpu().detach(), average='macro',multi_class='ovr')
            else: auc= roc_auc_score(target.cpu().detach(),torch.softmax(outputs,dim=1).cpu().detach(), average='macro',multi_class='ovr')
        else: auc= roc_auc_score(target.cpu().detach(),torch.sigmoid(outputs).cpu().detach(), average='macro',multi_class='ovr')


        if self.task == 'node_m':

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)

        f1 = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')
        return (f1,auc)


class MyVGAE(MyGAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(MyVGAE, self).__init__(encoder, decoder=decoder)


    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logvar__ = self.encoder(*args, **kwargs)
        self.__logvar__ = torch.clamp(self.__logvar__,min=LOG_VAR_MIN,max=LOG_VAR_MAX)

        z = self.reparametrize(self.__mu__, self.__logvar__)
        return z

    def kl_loss(self, mu=None, logvar=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar

        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)

    n_classes = len(classes)

    n_query = n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])


    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()


    return loss_val, prototypes

def proto_loss_qry(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0]/n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    return loss_val

def proto_qry_result(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0]/n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    _, y_hat = log_p_y.max(2)


    return y_hat


class GPDecoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(GPDecoder, self).__init__()
        self.weight = Parameter(torch.Tensor(1,in_channels))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph,z, edge_index = None, sigmoid=True):

        if graph.task == 'link':
            value = F.cosine_similarity(z[edge_index[0]],z[edge_index[1]],dim=-1)
            return torch.sigmoid(value) if sigmoid else value
        centroids = []
        z=z*self.weight
        for i in range(graph.num_classes):
            lab = graph.labels[graph.train_labels]
            if graph.task == 'node':tmp = (lab == i)
            if graph.task == 'node_m':tmp = (lab[:,i]==1)
            if tmp.sum() == 0:centroids.append(z[graph.train_labels].mean(0))
            else: centroids.append(z[graph.train_labels][tmp].mean(0))
            c = torch.vstack(centroids).to(z)

        z = z.unsqueeze(1)
        c = c.unsqueeze(0)

        distance = F.pairwise_distance(z,c)
        distance = 1/F.normalize(distance, dim=1)
        return distance

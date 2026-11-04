import torch
import math
from utils import *

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE, GINConv, SAGEConv
from torch.nn import Parameter,Linear,ModuleList
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.distributions import Normal
from torch import dropout, nn
import torch.nn.functional as F
from utils import uniform,to_torch_coo_tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from typing import Callable, Optional, Union,List,Tuple


from torch_sparse import SparseTensor, matmul


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class MetaGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(MetaGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0


        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, weight, bias, edge_weight=None, gamma=None, beta=None):
        """"""
        x = torch.matmul(x, weight)+bias


        if gamma is not None and beta is not None:
            x = torch.mul(x, gamma) + beta

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.Dropout = nn.Dropout(p=0.0)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)


    def forward(self, x):

        return (self.fc2(F.relu(self.Dropout(self.fc1(x)))))

    def reset_parameters(self):
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)


class MyGCNConv(MessagePassing):


    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(MyGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        mask = row == col

        norm = deg_inv_sqrt[row] * edge_weight
        norm[mask] = norm[mask]+1


        return edge_index, norm

    def forward(self, x, edge_index, edge_weight=None):
        """"""


        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result


        if edge_index.shape[1]>x.shape[0]**2*0.01:


            adj = torch.zeros(x.shape[0],x.shape[0]).to(norm)
            adj[edge_index[0],edge_index[1]] = norm
            return self.update(adj@x,bias=self.bias)


        else: return self.propagate(edge_index, x=x, norm=norm, bias=self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    def update(self, aggr_out, bias):
        aggr_out = aggr_out @ self.weight
        if self.bias is not None:
            aggr_out = aggr_out + bias
        return aggr_out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MetaSageConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(MetaSageConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None


        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0


        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight

    def forward(self, x, edge_index, weight, bias,edge_weight=None):
        """"""


        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm


        edge_index, norm = self.cached_result

        x = self.propagate(edge_index, x=x, norm=norm)


        return torch.matmul(x, weight)+bias

    def message(self,x_i, x_j, norm):
        feat = torch.cat([x_i, x_j], dim=-1)
        return norm.view(-1, 1) * feat

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class LampGCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 gating=None,
                 **kwargs):
        super(LampGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.gating = gating


        self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_2 = Parameter(torch.Tensor(1, 1))
        self.gating_weights = Parameter(torch.Tensor(in_channels, out_channels))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)


    def reset_parameters(self):
        glorot(self.weight_1)
        zeros(self.bias)
        glorot(self.gating_weights)


        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, weight_1, weight_2, bias, gating_weights, edge_weight=None, gamma=None, beta=None,parameters = False):
        """"""


        if gamma is not None and beta is not None:


            alpha = torch.sigmoid(torch.matmul(x, gating_weights).mean(0))
            gamma = torch.mul(alpha, gamma) + torch.mul(1 - alpha, torch.ones_like(gamma))
            beta = torch.mul(alpha, beta) + torch.mul(1 - alpha, torch.ones_like(beta))


        weight_1 = torch.mul(weight_1, gamma)
        bias = torch.mul(bias, beta)

        if parameters:
            self.weight_new = weight_1
            self.bias_new = bias


        x =  torch.matmul(x, weight_1)+bias

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return  self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MetaGatedGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 gating=None,
                 **kwargs):
        super(MetaGatedGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.gating = gating
        if gating == 'signature':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(1, 1))
            self.gating_weights = Parameter(torch.Tensor(out_channels))

        elif gating == 'weights':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(in_channels, out_channels))
            self.gating_weights = Parameter(torch.Tensor(out_channels))

        elif gating == 'signature_cond':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(1, 1))
            self.gating_weights = Parameter(torch.Tensor(in_channels, out_channels))

        elif gating == 'weights_cond':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(in_channels, out_channels))
            self.gating_weights = Parameter(torch.Tensor(in_channels, out_channels))

        else:
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(1, 1))
            self.gating_weights = Parameter(torch.Tensor(1, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_1)
        glorot(self.weight_2)
        zeros(self.bias)

        if self.gating.endswith('cond'):
            glorot(self.gating_weights)
        else:
            zeros(self.gating_weights)

        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, weight_1, weight_2, bias, gating_weights, edge_weight=None, gamma=None, beta=None):
        """"""

        if gamma is not None and beta is not None:
            if self.gating == 'signature':
                alpha = torch.sigmoid(gating_weights)
                gamma = torch.mul(alpha, gamma) + torch.mul(1 - alpha, torch.ones_like(gamma))
                beta = torch.mul(alpha, beta) + torch.mul(1 - alpha, torch.ones_like(beta))

                weight_1 = torch.mul(weight_1, gamma) + beta
                x = torch.matmul(x, weight_1)


            elif self.gating == 'weights':
                alpha = torch.sigmoid(gating_weights)

                x_1 = torch.matmul(x, weight_1)
                x_2 = torch.matmul(x, weight_2)
                x_2 = torch.mul(x_2, gamma) + beta

                x = torch.mul(alpha, x_1) + torch.mul(1 - alpha, x_2)

            elif self.gating == 'signature_cond':
                alpha = torch.sigmoid(torch.matmul(x, gating_weights))
                gamma = torch.mul(alpha, gamma) + torch.mul(1 - alpha, torch.ones_like(gamma))
                beta = torch.mul(alpha, beta) + torch.mul(1 - alpha, torch.ones_like(beta))


                x = torch.matmul(x, weight_1)
                x = torch.mul(x, gamma) + beta


            elif self.gating == 'weights_cond':
                alpha = torch.sigmoid(torch.matmul(x, gating_weights))

                x_1 = torch.matmul(x, weight_1)
                x_2 = torch.matmul(x, weight_2)
                x_2 = torch.mul(x_2, gamma) + beta

                x = torch.mul(alpha, x_1) + torch.mul(1 - alpha, x_2)

            else:
                x = torch.matmul(x, weight_1)
                x = torch.mul(x, gamma) + beta

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm, bias=bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, bias):
        if self.bias is not None:
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class MetaGRUCell(nn.Module):

    """
    An implementation of GRUCell with Functional Ops.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(MetaGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, weights, keys):
        x = x.view(-1, x.size(1))
        gate_x = F.linear(x,weights[keys[1]],weights[keys[2]])
        gate_h = F.linear(x,weights[keys[3]],weights[keys[4]])
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class MetaGatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}
        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}
        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})
    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super(MetaGatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Parameter(torch.Tensor(num_layers, out_channels, out_channels))
        self.rnn = MetaGRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, weights, keys, edge_weight=None):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, weights[keys[0]][i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            h = self.rnn(m, h, weights, keys)

        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)


class MetaGINConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 eps = 0.,
                 train_eps = True,
                 **kwargs):
        super(MetaGINConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.train_eps = train_eps
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0


        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight

    def forward(self, x, edge_index, eps = 0,edge_weight=None):
        """"""


        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        if self.train_eps:
            out = (1 + eps)* x +  self.propagate(edge_index, x=x, norm=norm,)
        else:
            out = (1 + self.eps)* x +  self.propagate(edge_index, x=x, norm=norm)

        return torch.matmul(out, self.weight)+self.bias

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)














class SemanticSpaceSAGEConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        topk: int = 5,
        tau: float = 0.5,
        lamda:float= 0.5,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super(SemanticSpaceSAGEConv, self).__init__(**kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        self.topk = topk
        self.tau = tau
        self.lamda = lamda


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.factor_neighbor_projection = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
            self.factor_root_projection = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.factor_neighbor_projection.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
            self.factor_root_projection.reset_parameters()

    def forward(
        self,
        xes: List[Union[torch.Tensor, OptPairTensor]],
        edge_index: Adj,
        semantic_factors: Union[torch.Tensor, OptPairTensor],
        factor_edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        xs = []
        for i in range(len(xes)):
            if isinstance(xes[i], Tensor):
                xs.append((xes[i], xes[i]))

        if isinstance(semantic_factors, Tensor):
            semantic_factors = (semantic_factors, semantic_factors)


        degrees = factor_edge_index.sum(dim=1, keepdim=True)
        weighted_sum = factor_edge_index @ semantic_factors[0]
        factor_output = weighted_sum / (degrees + 1e-9)


        outs = []
        for i in range(len(xs)):
            out = self.propagate(edge_index, x=xs[i][0], size=size)

            similarity = xs[i][0] @ semantic_factors[0].T
            weights,_ = gumbel_topk(similarity, self.topk, self.tau)

            out = self.lamda*out + (1-self.lamda)*(weights @ semantic_factors[0])
            outs.append(self.lin_l(out))

        factor_output = self.factor_neighbor_projection(factor_output)


        for i in range(len(xs)):
            x_r = xs[i][1]
            if self.root_weight and x_r is not None:
                outs[i] = outs[i] + self.lin_r(x_r)

            if self.normalize:
                outs[i] = F.normalize(outs[i], p=2., dim=-1)


        root_factors = semantic_factors[1]
        if self.root_weight and root_factors is not None:
            factor_output = factor_output + self.factor_root_projection(root_factors)
        if self.normalize:
            factor_output = F.normalize(factor_output, p=2., dim=-1)

        return outs,factor_output


    def naive_forward(
        self,
        xes: List[Union[torch.Tensor, OptPairTensor]],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        xs = []
        for i in range(len(xes)):
            if isinstance(xes[i], Tensor):
                xs.append((xes[i], xes[i]))


        outs = []
        for i in range(len(xs)):
            out = self.propagate(edge_index, x=xs[i][0], size=size)
            outs.append(self.lin_l(out))


        for i in range(len(xs)):
            x_r = xs[i][1]
            if self.root_weight and x_r is not None:
                outs[i] += self.lin_r(x_r)

            if self.normalize:
                outs[i] = F.normalize(outs[i], p=2., dim=-1)


        return outs[0],outs[1],outs[2]


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SemanticFusedSAGEConv(nn.Module):
    """Semantic retrieval, modality fusion and GraphSAGE propagation, Eqs. (9)--(14)."""
    def __init__(self, in_channels, out_channels, topk=5, tau=0.5,
                 gamma=0.5, modals=3, eta=0.25, **kwargs):
        super().__init__()
        self.topk, self.tau, self.gamma, self.eta = topk, tau, gamma, eta
        self.modals = modals
        self.query_projections = ModuleList([MLP(in_channels, in_channels, in_channels)
                                         for _ in range(modals)])
        self.semantic_projections = ModuleList([MLP(in_channels, in_channels, in_channels)
                                       for _ in range(modals)])
        # f_g^l is shared across modalities, each propagated separately.
        self.conv = SAGEConv(in_channels, out_channels, aggr='mean')
        self.grounding_loss = None

    def reset_parameters(self):
        self.conv.reset_parameters()
        for module in list(self.query_projections)+list(self.semantic_projections):
            module.reset_parameters()

    def forward(self, xes, edge_index, semantic_factors, factor_edge_index=None,
                mask=None, compute_grounding=True):
        logits, selections, semantics = [], [], []
        grounding = semantic_factors.sum()*0.0
        for m, x in enumerate(xes):
            query = self.query_projections[m](x)
            score = query @ semantic_factors.t()                 # Eq. (9)
            weight, _, selection = gumbel_topk(
                score, self.topk, self.tau, self.training, return_mask=True)
            logits.append(score); selections.append(selection)
            semantics.append(weight @ semantic_factors)
            if compute_grounding:
                grounding = grounding + semantic_grounding_loss(
                    query, semantic_factors, weight,
                    None if mask is None else mask[:, m], self.eta)
        if mask is not None:                               # only initial stage
            scores = torch.stack(logits, dim=1)             # N,M,T
            selected = torch.stack(selections, dim=1) * mask.unsqueeze(-1)
            joint = selected_softmax(scores.flatten(1), selected.flatten(1))
            joint = joint.reshape_as(scores)
            # The same factor selected by two modalities contributes twice.
            shared = joint.sum(dim=1) @ semantic_factors
            semantics = [mask[:, m:m+1]*p + (1-mask[:, m:m+1])*shared
                         for m, p in enumerate(semantics)]
        outputs = []
        for m, (x, semantic) in enumerate(zip(xes, semantics)):
            fused = self.gamma*self.semantic_projections[m](semantic) + (1-self.gamma)*x
            outputs.append(self.conv(fused, edge_index))
        self.grounding_loss = grounding
        return outputs, semantic_factors

    def naive_forward(self, xes, edge_index):
        return [self.conv(x, edge_index) for x in xes]


class SemanticSpaceSAGEConvL1(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        topk: int = 50,
        tau: float = 0.5,
        lamda:float= 0.5,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super(SemanticSpaceSAGEConvL1, self).__init__(**kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        self.topk = topk
        self.tau = tau
        self.lamda = lamda


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        xes: List[Union[torch.Tensor, OptPairTensor]],
        edge_index: Adj,
        semantic_factors: Union[torch.Tensor, OptPairTensor],
        factor_edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        xs = []
        for i in range(len(xes)):
            if isinstance(xes[i], Tensor):
                xs.append((xes[i], xes[i]))

        if isinstance(semantic_factors, Tensor):
            semantic_factors = (semantic_factors, semantic_factors)


        degrees = factor_edge_index.sum(dim=1, keepdim=True)
        weighted_sum = factor_edge_index @ semantic_factors[0]
        factor_output = weighted_sum / (degrees + 1e-9)


        outs = []
        for i in range(len(xs)):
            out = self.propagate(edge_index, x=xs[i][0], size=size)

            similarity = xs[i][0] @ semantic_factors[0].T
            weights,_ = gumbel_topk(similarity, self.topk, self.tau)

            out = self.lamda*out + (1-self.lamda)*(weights @ semantic_factors[0])
            outs.append(self.lin_l(out))

        factor_output = self.lin_l(factor_output)


        for i in range(len(xs)):
            x_r = xs[i][1]
            if self.root_weight and x_r is not None:
                outs[i] = outs[i] + self.lin_r(x_r)

            if self.normalize:
                outs[i] = F.normalize(outs[i], p=2., dim=-1)


        root_factors = semantic_factors[1]
        if self.root_weight and root_factors is not None:
            factor_output = factor_output + self.lin_r(root_factors)
        if self.normalize:
            factor_output = F.normalize(factor_output, p=2., dim=-1)

        return (outs[0]+outs[1]+outs[2])/3,factor_output


    def naive_forward(
        self,
        xes: List[Union[torch.Tensor, OptPairTensor]],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        xs = []
        for i in range(len(xes)):
            if isinstance(xes[i], Tensor):
                xs.append((xes[i], xes[i]))


        outs = []
        for i in range(len(xs)):
            out = self.propagate(edge_index, x=xs[i][0], size=size)
            outs.append(self.lin_l(out))


        for i in range(len(xs)):
            x_r = xs[i][1]
            if self.root_weight and x_r is not None:
                outs[i] += self.lin_r(x_r)

            if self.normalize:
                outs[i] = F.normalize(outs[i], p=2., dim=-1)


        return outs[0],outs[1],outs[2]


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SemanticSpaceSAGEConvL2(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        topk: int = 50,
        tau: float = 0.5,
        lamda:float= 0.5,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super(SemanticSpaceSAGEConvL2, self).__init__(**kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        self.topk = topk
        self.tau = tau
        self.lamda = lamda


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        semantic_factors: Union[Tensor, OptPairTensor],
        factor_edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)
        if isinstance(semantic_factors, Tensor):
            semantic_factors = (semantic_factors, semantic_factors)


        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        degrees = factor_edge_index.sum(dim=1, keepdim=True)
        weighted_sum = factor_edge_index @ semantic_factors[0]
        factor_output = weighted_sum / (degrees + 1e-9)


        similarity = x[0] @ semantic_factors[0].T
        weights,_ = gumbel_topk(similarity, self.topk, self.tau)
        out = self.lamda*out + (1-self.lamda)*(weights @ semantic_factors[0])


        out = self.lin_l(out)
        factor_output = self.lin_l(factor_output)


        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)


        root_factors = semantic_factors[1]
        if self.root_weight and root_factors is not None:
            factor_output = factor_output + self.lin_r(root_factors)
        if self.normalize:
            factor_output = F.normalize(factor_output, p=2., dim=-1)

        return out,factor_output


    def naive_forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)


        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)


        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)


        return out,out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t,x) -> Tensor:


        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def weighted_semantic_grounding(node_features, semantic_factors,top_k,mask = None, eta=0.25):

    similarity = torch.sigmoid(node_features @ semantic_factors.T)  # Shape [N, T].
    top_k_sim, top_k_indices = similarity.topk(top_k, dim=-1, largest=True, sorted=False)  # Shape [N, k].

    node_features_stop = node_features.detach()  # stop-gradient
    selected_factors = semantic_factors[top_k_indices]

    factor_alignment_loss = torch.sum(top_k_sim*torch.norm(node_features_stop.unsqueeze(1)- selected_factors, p=2, dim=-1), dim=-1)


    if mask is not None:
        factor_alignment_loss = factor_alignment_loss@mask/torch.sum(mask) # Average over observed nodes.

    else:factor_alignment_loss =  torch.mean(factor_alignment_loss)

    detached_factors = semantic_factors.detach()  # stop-gradient
    detached_selected_factors = detached_factors[top_k_indices]
    commitment_loss = torch.sum(top_k_sim*torch.norm(node_features.unsqueeze(1)- detached_selected_factors, p=2, dim=-1), dim=-1)

    if mask is not None:
        commitment_loss = commitment_loss@mask/torch.sum(mask) # Average over observed nodes.

    else:commitment_loss =  torch.mean(commitment_loss)

    # Combine factor alignment with the weighted commitment loss.
    loss = factor_alignment_loss + eta * commitment_loss

    return loss

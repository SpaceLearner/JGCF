from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import sum, mul, fill_diag, remove_diag
from torch.nn import Parameter

import numpy as np
import time

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


class GeneralPropagation(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 add_self_loops_l1: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 num_classes: int = None,
                 args=None,
                 **kwargs):

        super(GeneralPropagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout = args.prop_dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.add_self_loops_l1 = add_self_loops_l1

        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

        self.node_num = node_num
        self.num_classes = num_classes
        self.args = args
        self.max_value = None
        self.debug = self.args.debug

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

    def get_incident_matrix(self, edge_index: Adj):
        size = edge_index.sizes()[1]
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        mask = row_index >= col_index
        row_index = row_index[mask]
        col_index = col_index[mask]
        edge_num = row_index.numel()
        row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
        col = torch.cat([row_index, col_index])
        value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
        inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(edge_num, size))
        return inc

    def inc_norm(self, inc, edge_index, add_self_loops):
        if add_self_loops:
            edge_index = fill_diag(edge_index, 1.0)
        else:
            edge_index = remove_diag(edge_index)
        deg = sum(edge_index, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        inc = mul(inc, deg_inv_sqrt.view(1, -1))  ## col-wise
        return inc

    def check_inc(self, edge_index, inc, normalize=False):
        # return None  ## not checking it
        nnz = edge_index.nnz()
        if normalize:
            deg = torch.eye(edge_index.sizes()[0])  # .cuda()
        else:
            deg = sum(edge_index, dim=1).cpu()
            deg = torch.diag(deg)
        inc = inc.cpu()
        lap = (inc.t() @ inc).to_dense()
        adj = edge_index.cpu().to_dense()

        lap2 = deg - adj
        diff = torch.sum(torch.abs(lap2 - lap)) / nnz
        # import ipdb; ipdb.set_trace()
        assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'

    def forward(self, x: Tensor, edge_index: Adj, x_idx: Tensor = None,
                edge_weight: OptTensor = None, mode=None, niter=None,
                data=None) -> Tensor:
        """"""
        start_time = time.time()
        edge_index2 = edge_index
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                ## first cache incident_matrix (before normalizing edge_index)
                # print("here ******************************************************** ")
                cache = self._cached_inc
                if cache is None:
                    incident_matrix = self.get_incident_matrix(edge_index=edge_index)
                    # print("here ******************************************************** ")
                    # if not self.args.ogb:
                    #     self.check_inc(edge_index=edge_index, inc=incident_matrix, normalize=False)
                    # print("here1 ******************************************************** ")
                    incident_matrix = self.inc_norm(inc=incident_matrix, edge_index=edge_index,
                                                    add_self_loops=self.add_self_loops_l1)
                    # print("here ******************************************************** ")
                    if not self.args.ogb:
                        edge_index_C = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), False,
                            add_self_loops=self.add_self_loops_l1, dtype=x.dtype)
                        # self.check_inc(edge_index=edge_index_C, inc=incident_matrix, normalize=True)
                    
                    # print("here ******************************************************** ")

                    if self.cached:
                        self._cached_inc = incident_matrix
                        self.init_z = torch.zeros((incident_matrix.sizes()[0], x.size()[-1])).cuda()
                else:
                    incident_matrix = self._cached_inc
                
                # print("here ******************************************************** ")

                cache = self._cached_adj_t
                if cache is None:
                    # if True:
                    if False:
                        edge_index = self.doubly_stochastic_norm(edge_index, x, self.add_self_loops)  ##
                    else:
                        edge_index = gcn_norm(
                            edge_index, edge_weight, x.size(self.node_dim), False,
                            add_self_loops=self.add_self_loops, dtype=x.dtype)

                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        K_ = self.K if niter is None else niter
        if mode == None: mode = self.mode
        assert edge_weight is None
        if K_ <= 0:
            return x

        hh = x

        x, xs = self.gtn_forward(x=x, hh=hh, incident_matrix=incident_matrix, K=K_)
        return x, xs

    def gtn_forward(self, x, hh, K, incident_matrix):
        lambda2 = self.args.lambda2
        beta = self.args.beta
        gamma = None

        ############################# parameter setting ##########################
        if gamma is None:
            gamma = 1

        if beta is None:
            beta = 1 / 2

        if lambda2 > 0: z = self.init_z.detach()

        xs = []
        for k in range(K):
            grad = x - hh
            smoo = x - gamma * grad
            temp = z + beta / gamma * (incident_matrix @ (smoo - gamma * (incident_matrix.t() @ z)))

            z = self.proximal_l1_conjugate(x=temp, lambda2=lambda2, beta=beta, gamma=gamma, m="L1")
            # import ipdb; ipdb.set_trace()

            ctz = incident_matrix.t() @ z

            x = smoo - gamma * ctz

            x = F.dropout(x, p=self.dropout, training=self.training)

        # print("wihtout average")
        light_out = x

        return light_out, xs

    def proximal_l1_conjugate(self, x: Tensor, lambda2, beta, gamma, m):
        if m == 'L1':
            x_pre = x
            x = torch.clamp(x, min=-lambda2, max=lambda2)
            # print('diff after proximal: ', (x-x_pre).norm())

        elif m == 'L1_original':  ## through conjugate
            rr = gamma / beta
            yy = rr * x
            x_pre = x
            temp = torch.sign(yy) * torch.clamp(torch.abs(yy) - rr * lambda2, min=0)
            x = x - temp / rr

        else:
            raise ValueError('wrong prox')
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={})'.format(self.__class__.__name__, self.K,
                                                                self.alpha, self.mode, self.dropout)
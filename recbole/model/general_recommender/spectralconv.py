# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, transpose
from torch_sparse import sum as sparsesum

from torch_scatter import scatter_add, scatter_softmax, scatter_sum, scatter_min, scatter_max
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tqdm import tqdm

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
from torch import lobpcg

class SpectralConv(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SpectralConv, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        self.k = config["k"]
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.virt_embedding = torch.nn.Embedding(
            num_embeddings=config["k"], embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.spectral_adj_matrix_from, self.spectral_adj_matrix_to = self.get_spectral_adj_matrix()# .to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        self.adj_matrix = L
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        # SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        
        SparseL = SparseTensor(row=i[0], col=i[1], value=data, sparse_sizes=(self.n_users+self.n_items, self.n_users+self.n_items)) # torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        # SparseL = gcn_norm(SparseL, num_nodes=self.n_users+self.n_items)
        
        return SparseL
    
    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x"""
        import faiss

        kmeans = faiss.Kmeans(d=256, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster
    
    def get_spectral_adj_matrix(self):
        
        # edge_index = coo_matrix((np.ones(self.adj_matrix.nnz), (self.adj_matrix.row, self.adj_matrix.col)), shape=(self.n_users+self.n_items, self.n_users+self.n_items))
        row = self.adj_matrix.row
        col = self.adj_matrix.col
        i = torch.LongTensor(np.array([row, col])).to(self.device)
        data = torch.FloatTensor(self.adj_matrix.data).to(self.device)
        A = torch.sparse.FloatTensor(i, data, torch.Size(self.adj_matrix.shape))
        values, vecs = lobpcg(A, k=256, largest=True)
        centroids, node2cluster = self.run_kmeans(vecs.detach().cpu().numpy())
        hyper_edge_index = torch.stack([torch.arange(node2cluster.shape[0], device=self.device), node2cluster])
        adj_mat = coo_matrix((np.ones_like(node2cluster.cpu().detach().numpy()), (hyper_edge_index[0].cpu().detach().numpy(), hyper_edge_index[1].cpu().detach().numpy())), shape=(self.n_users+self.n_items, self.k)).tocsr()
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocoo()
        
        SparseL1 = SparseTensor(row=torch.tensor(norm_adj.row, dtype=torch.long, device=self.device), col=torch.tensor(norm_adj.col, dtype=torch.long, device=self.device), value=torch.tensor(norm_adj.data, device=self.device), sparse_sizes=(self.n_users+self.n_items, self.k))
        SparseL2 = SparseTensor(row=torch.tensor(norm_adj.col, dtype=torch.long, device=self.device), col=torch.tensor(norm_adj.row, dtype=torch.long, device=self.device), value=torch.tensor(norm_adj.data, device=self.device), sparse_sizes=(self.k, self.n_users+self.n_items))
        return SparseL1, SparseL2
        

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        virt_embeddings = self.virt_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, virt_embeddings

    def forward(self):
        all_embeddings, virt_embeddings = self.get_ego_embeddings()
        all_embeddings_g = matmul(self.spectral_adj_matrix_from, virt_embeddings)
        all_embeddings  = all_embeddings + all_embeddings_g
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings_g = matmul(self.spectral_adj_matrix_from, virt_embeddings)
            all_embeddings = matmul(self.norm_adj_matrix, all_embeddings + all_embeddings_g)
            
            # virt_embeddings = matmul(self.spectral_adj_matrix_to, all_embeddings_g)
            
            embeddings_list.append(all_embeddings)
            
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

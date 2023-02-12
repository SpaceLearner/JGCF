import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

import gc

class GDE(GeneralRecommender):
    
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset) -> None:
        super(GDE, self).__init__(config, dataset)
        
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.get_norm_adj_mat()
        
        self.latent_dim = config["embedding_size"]
        
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        
        self.beta=config["beta"]
		self.reg=config["reg_weight"]
		self.drop_out=config["dropout_rate"]
		if self.drop_out!=0:
			self.m=torch.nn.Dropout(self.drop_out)

		if config["feature_type"]=='smoothed':
			user_filter=self.weight_feature(torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_user_values.npy')).cuda())
			item_filter=self.weight_feature(torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_item_values.npy')).cuda())

			user_vector=torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_user_features.npy')).cuda()
			item_vector=torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_item_features.npy')).cuda()


		elif config["feature_type"]=='both':

			user_filter=torch.cat([self.weight_feature(torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_user_values.npy')).cuda())\
				,self.weight_feature(torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_rough_user_values.npy')).cuda())])

			item_filter=torch.cat([self.weight_feature(torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_item_values.npy')).cuda())\
				,self.weight_feature(torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_rough_item_values.npy')).cuda())])


			user_vector=torch.cat([torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_user_features.npy')).cuda(),\
				torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_rough_user_features.npy')).cuda()],1)


			item_vector=torch.cat([torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_smooth_item_features.npy')).cuda(),\
				torch.Tensor(np.load(r'./'+dataset+ r'/'+dataset+'_rough_item_features.npy')).cuda()],1)


		else:
			print('error')
			exit()

		self.L_u=(user_vector*user_filter).mm(user_vector.t())
		self.L_i=(item_vector*item_filter).mm(item_vector.t())


		del user_vector,item_vector,user_filter, item_filter
		gc.collect()
		torch.cuda.empty_cache()
        
        
        
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
            (self.n_users, self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col), [1] * inter_M.nnz)
        )
        
        A._update(data_dict)
        
        adj_mat = A.tocoo()
        self.adj_mat = adj_mat.tocsr()
        # norm adj matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = svds(self.norm_adj, 256)
        
    def forward(self):
        pass
    
    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        
        return torch.nn.Parameter(torch.zeros(1))
    
    def predict(self, interaction):
        
        pass

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].detach().cpu().numpy()
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[user,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        # if(self.dataset.name == 'amazon-book'):
        #     ret = U_2
        # else:
        U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
        ret = U_2 + 0.3 * U_1
        
        ret = torch.tensor(ret)
        return ret.view(ret.shape[0], -1)
    
# class GF_CF(object):
#     def __init__(self, adj_mat):
#         self.adj_mat = adj_mat
        
#     def train(self):
#         adj_mat = self.adj_mat
#         start = time.time()
#         rowsum = np.array(adj_mat.sum(axis=1))
#         d_inv = np.power(rowsum, -0.5).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat = sp.diags(d_inv)
#         norm_adj = d_mat.dot(adj_mat)

#         colsum = np.array(adj_mat.sum(axis=0))
#         d_inv = np.power(colsum, -0.5).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat = sp.diags(d_inv)
#         self.d_mat_i = d_mat
#         self.d_mat_i_inv = sp.diags(1/d_inv)
#         norm_adj = norm_adj.dot(d_mat)
#         self.norm_adj = norm_adj.tocsc()
#         ut, s, self.vt = sparsesvd(self.norm_adj, 256)
#         end = time.time()
#         print('training time for GF-CF', end-start)
        
#     def getUsersRating(self, batch_users, ds_name):
#         norm_adj = self.norm_adj
#         adj_mat = self.adj_mat
#         batch_test = np.array(adj_mat[batch_users,:].todense())
#         U_2 = batch_test @ norm_adj.T @ norm_adj
#         if(ds_name == 'amazon-book'):
#             ret = U_2
#         else:
#             U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
#             ret = U_2 + 0.3 * U_1
#         return ret
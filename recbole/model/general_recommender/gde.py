import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from recbole.model.loss import BPRLoss, EmbLoss

import gc

def cal_spectral_feature(Adj, size, type='user', largest=True, niter=5):
    
    value, vector=torch.lobpcg(Adj,k=size, largest=largest,niter=niter)
    
    return value, vector

class GDE(GeneralRecommender):
    
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset) -> None:
        super(GDE, self).__init__(config, dataset)
        
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        
        self.latent_dim = config["embedding_size"]
        
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        
        self.adaptive = config["adaptive"]
        
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        self.beta          = config["beta"]
        self.reg           = config["reg_weight"]
        self.drop_out      = config["dropout_rate"]
        self.smooth_ratio  = config["smooth_ratio"]
        self.rough_ratio   = config["rough_ratio"]
        self.require_pow   = config["require_pow"]
        self.reg_weight    = config["reg_weight"]
        
        if self.drop_out!=0:
            self.m=torch.nn.Dropout(self.drop_out)
            
        self.Adj = self.get_norm_adj_mat().to(self.device)
        
        if config["feature_type"]=='smoothed':
            
            user_adj  = self.Adj.matmul(self.Adj.T).clip(max=1.0)
            user_adj_index = user_adj.nonzero().T
            adj_value = user_adj[user_adj_index[0], user_adj_index[1]]
            # print(user_adj.shape)
            user_adj = torch.sparse_coo_tensor(indices=user_adj_index, values=adj_value)
            user_value, user_vec = cal_spectral_feature(Adj=user_adj, size=int(self.n_users*self.smooth_ratio))
            user_filter=self.weight_feature(user_value.to(self.device))
            user_vector=user_vec.to(self.device)
            
            item_adj  = self.Adj.T.matmul(self.Adj).clip(max=1.0)
            item_adj_index = item_adj.nonzero().T
            adj_value = item_adj[item_adj_index[0], item_adj_index[1]]
            
            item_adj = torch.sparse_coo_tensor(indices=item_adj_index, values=adj_value)
            item_value, item_vec = cal_spectral_feature(Adj=item_adj, size=int(self.n_items * self.smooth_ratio))
            item_filter=self.weight_feature(item_value.to(self.device))
            item_vector=item_vec.to(self.device)
            
        elif config["feature_type"]=='both':

            user_adj  = self.Adj.matmul(self.Adj.T).clip(max=1.0)
            user_adj_index = user_adj.nonzero().T
            adj_value = user_adj[user_adj_index[0], user_adj_index[1]]
            user_adj = torch.sparse_coo_tensor(indices=user_adj_index, values=adj_value)
            user_value, user_vec     = cal_spectral_feature(Adj=user_adj, size=int(self.n_users*self.smooth_ratio))
            user_value_r, user_vec_r = cal_spectral_feature(Adj=self.Adj.matmul(self.Adj.T).clip(max=1.0), size=int(self.n_users*self.rough_ratio), largest=False)
            
            user_filter=torch.cat([self.weight_feature(user_value.to(self.device)), self.weight_feature(user_value_r.to(self.device))])
            user_vector=torch.cat([user_vec.to(self.device), user_vec_r.to(self.device)],1)
            
            
            item_adj  = self.Adj.T.matmul(self.Adj).clip(max=1.0)
            item_adj_index = item_adj.nonzero().T
            adj_value = item_adj[item_adj_index[0], item_adj_index[1]]
            
            item_adj = torch.sparse_coo_tensor(indices=item_adj_index, values=adj_value)
            item_value, item_vec     = cal_spectral_feature(Adj=item_adj, size=int(self.n_users*self.smooth_ratio))
            item_value_r, item_vec_r = cal_spectral_feature(Adj=item_adj, size=int(self.n_users*self.rough_ratio), largest=False)
            
            item_filter=torch.cat([self.weight_feature(item_value.to(self.device)),self.weight_feature(item_value_r.to(self.device))])
            item_vector=torch.cat([item_vec, item_vec_r], dim=1)
        
        print(user_vector.shape)

        self.L_u=(user_vector*user_filter).mm(user_vector.t())
        self.L_i=(item_vector*item_filter).mm(item_vector.t())
        del user_vector,item_vector,user_filter, item_filter
        
        gc.collect()
        torch.cuda.empty_cache()
        
    def weight_feature(self,value):
        return torch.exp(self.beta*value)
        
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
        
        R = torch.tensor(A.todense())
        
        D_u=R.sum(1)
        D_i=R.sum(0)


        #in the case any users or items have no interactions
        for i in range(self.n_users):
            if D_u[i]!=0:
                D_u[i]=1/D_u[i].sqrt()

        for i in range(self.n_items):
            if D_i[i]!=0:
                D_i[i]=1/D_i[i].sqrt()
                
        Rn = D_u.unsqueeze(1)*R*D_i
        
        return Rn
        
    def forward(self):
        
        return self.L_u.mm(self.user_embedding.weight), self.L_i.mm(self.item_embedding.weight)
        # final_user,final_pos,final_nega=self.L_u[user].mm(self.user_embed.weight),self.L_i[pos_item].mm(self.item_embed.weight),self.L_i[nega_item].mm(self.item_embed.weight)
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # user_all_embeddings, item_all_embeddings = self.forward()
        # u_embeddings = user_all_embeddings[user]
        # pos_embeddings = item_all_embeddings[pos_item]
        # neg_embeddings = item_all_embeddings[neg_item]
        
        if self.drop_out == 0.0:
            u_embeddings   = self.L_u[user].mm(self.user_embedding.weight)
            pos_embeddings = self.L_i[pos_item].mm(self.item_embedding.weight)
            neg_embeddings = self.L_i[neg_item].mm(self.item_embedding.weight)
        else:
            u_embeddings   = (self.m(self.L_u[user])*(1-self.drop_out)).mm(self.user_embedding.weight)
            pos_embeddings = (self.m(self.L_i[pos_item])*(1-self.drop_out)).mm(self.item_embedding.weight)
            neg_embeddings = (self.m(self.L_i[neg_item])*(1-self.drop_out)).mm(self.item_embedding.weight)

        # calculate BPR Loss
        
        if self.adaptive:
            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            res_neg=(u_embeddings*neg_embeddings).sum(1)
            neg_weight=(1-(1-res_neg.sigmoid().clamp(max=0.99)).log10()).detach()
            neg_scores = neg_weight * res_neg
        else:
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
        
        pass

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
       #  print(user)
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
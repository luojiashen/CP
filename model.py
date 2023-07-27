import torch
from torch import nn
from math import ceil,floor
from dataset import dense_mat_to_coo_mat
from os import path
import numpy as np
from dataset import *
import scipy.sparse as sp
image_root='evolution_progress'

class User_Preference_Net(torch.nn.Module):
    def __init__(self,item_emb_dim,user_num,hidden_num):
        super().__init__()
        self.item_emb_dim=item_emb_dim
        self.user_num=user_num
        self.hidden_num=hidden_num
        self.net=self.get_user_net()
    def get_user_net(self):
        '''
        根据隐层数量生成对应的多层感知机
        '''
        item_user_gap=int((self.user_num-self.item_emb_dim)/(self.hidden_num+1))
        hiddens_dim=[(x+1)*item_user_gap for x in range(0,self.hidden_num)]
        net_dims=[self.item_emb_dim]+hiddens_dim+[self.user_num]
        net=torch.nn.Sequential()
        for i in range(0,2*(len(net_dims)-1),2):
            net.add_module('linear {}'.format(i),torch.nn.Linear(net_dims[floor(i/2)],net_dims[ceil((i+1)/2)]))
            if i==(2*(len(net_dims)-1)-1-((2*(len(net_dims)-1)-1)%2)):
                net.add_module('sigmoid {}'.format(i+1),torch.nn.Sigmoid())
            else:
                net.add_module('normalization {}'.format(i+1),torch.nn.LayerNorm(net_dims[ceil((i+1)/2)]))
                net.add_module('sigmoid {}'.format(i+1),torch.nn.Sigmoid())
        return net
    def forward(self,x):
        return self.net(x)

class CP(torch.nn.Module):
    def __init__(self,data:data_cp,args):
        super().__init__()
        self.embs_dim=args.embedding_dim
        self.item_num=data.item_num
        self.item_embs = None
        self.layer_num = args.layer_num
        self.args = args
        # UPN
        self.user_p_net=User_Preference_Net(self.embs_dim,data.user_num,args.hidden_num)
        
    
    def forward(self,item_idxes=None):

        output = self.user_p_net(self.item_embs[item_idxes])
        return output
    def predict(self,user,item):
        
        return  self.user_p_net(self.item_embs[item])[:,user]


class LightGCN(torch.nn.Module):
    def __init__(self,data,args):
        super(LightGCN,self).__init__()
        self.embs_dim=args.embedding_dim
        self.item_num=data.item_num
        self.user_num=data.user_num
        self.device=args.device
        self.layer_num=args.layer_num
        self.item_embs=torch.nn.Embedding(data.item_num,self.embs_dim)
        self.user_embs=torch.nn.Embedding(data.user_num,self.embs_dim)# user dim=item dim

        self.graph=data.get_symmetrically_normalized_matrix()
        self.L2_coefficient=args.L2_coefficient

    def gcn_layer_combination(self):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_emb = torch.cat([users_emb, items_emb]) #在零维对用户物品向量进行拼接

        embs = [all_emb]
        
        for layer in range(self.layer_num):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_num, self.item_num])
        return users, items
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.gcn_layer_combination()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embs(users)
        pos_emb_ego = self.item_embs(pos_items)
        neg_emb_ego = self.item_embs(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def loss(self,users,pos,neg):
        '''bpr loss , regularization loss'''
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # softplus为Relu函数的平滑近似，可用于控制模型的输出保持为正
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    def forward(self,users,items):
        # compute embedding
        all_users, all_items = self.gcn_layer_combination()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb) #向量哈达玛积
        predicts  = torch.sum(inner_pro, dim=1)
        return predicts
    def predict(self,users,items):
        return self.forward(users,items)

class Positive_Fit_Loss(torch.nn.Module):
    def __init__(self,individuality=0.4) -> None:
        super().__init__()
        self.individuality=individuality 
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,y_pred,y_true):
        ng_rate=self.individuality*torch.sum(y_true,dim=0)/torch.sum(y_true+(1-y_true),dim=0)
        y_false=1-y_true
        rand=torch.rand_like(y_true)
        mask=rand>ng_rate
        y_false[mask]=0
        return torch.sum(-1*y_true*torch.log(y_pred)-y_false*torch.log(1-y_pred))
    
class BPR(torch.nn.Module):
    def __init__(self, data, args):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embs_dim: number of predictive factors.
        """
        user_num=data.user_num
        item_num=data.item_num
        self.embs_dim=args.embedding_dim
        self.user_embs = torch.nn.Embedding(user_num, self.embs_dim)
        self.item_embs = torch.nn.Embedding(item_num, self.embs_dim)

        torch.nn.init.normal_(self.user_embs.weight, std=0.01)
        torch.nn.init.normal_(self.item_embs.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.user_embs(user)
        item_i = self.item_embs(item_i)
        item_j = self.item_embs(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j
    def predict(self,user,item):
        users=self.user_embs.weight[user]# (user num,emb dim)
        items=self.item_embs.weight[item]
        predicts=(users * items).sum(dim=-1)
        return predicts
model_dict={'bpr':BPR,
            'lightgcn':LightGCN,
            'cp':CP}

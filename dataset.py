import os
from time import time
import torch
import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from math import ceil,floor
from torch.utils.data import Dataset,DataLoader
from cprint import c_print
import pickle
import copy
from scipy.sparse import csr_matrix
import scipy.sparse as sp

class data_basic:
    def __init__(self,args,
                 model_name='basic',
                 use_cols=None):
        self.data_split_rate=args.split_rate
        self.data_name=args.data_name
        self.device=args.device
        self.args=args
        self.data_root=os.path.join('data',args.data_name)
        print("Dataset: " + args.data_name)
        
        if use_cols:
            self.data_df=pd.read_csv(os.path.join(self.data_root,'sm.csv'),usecols=use_cols)
        else:
            self.data_df=pd.read_csv(os.path.join(self.data_root,'sm.csv'))
        self.Info()
        
        if args.data_preprocess:
            self.data_split()
    
    def Info(self):
        self.num_rating=len(self.data_df)
        self.user_num=len(self.data_df['userid'].value_counts().keys())
        self.item_num=len(self.data_df['itemid'].value_counts().keys())
        print("[Interaction num]:{}\n[User num]:{}\n[Item num]:{}".format(
            self.num_rating,self.user_num,self.item_num))
        print(f"Shape of Super Matrix:{self.data_df.shape}")
        c_print("sparsity:{}".format(self.num_rating/(self.user_num*self.item_num)))
    
    def data_split(self):
        user_item_dict=dict()
        for idx in self.data_df.index:
            if self.data_df.loc[idx,'userid'] not in user_item_dict.keys():
                user_item_dict[self.data_df.loc[idx,'userid']]=\
                    [self.data_df.loc[idx,'itemid']]
            else:
                user_item_dict[self.data_df.loc[idx,'userid']].append(
                    self.data_df.loc[idx,'itemid'])

        train,test,valid=self.data_split_rate[0],self.data_split_rate[2],self.data_split_rate[1]
        ui_train=dict()
        ui_test=dict()
        ui_valid=dict()
        for k,v in tqdm.tqdm(user_item_dict.items(),leave=False):
            l=len(v)
            v=np.array(v)
            np.random.shuffle(v)
            if len(v[:ceil(train*l)])!=0:
                ui_train[k]=v[:ceil(train*l)]
            if len(v[ceil(train*l):ceil((train+test)*l)])!=0:
                ui_test[k]=v[ceil(train*l):ceil((train+test)*l)]
            if len(v[ceil((train+test)*l):])!=0:
                ui_valid[k]=v[ceil((train+test)*l):]
        ui_train=self.dict_to_df(ui_train)
        ui_test=self.dict_to_df(ui_test)
        ui_valid=self.dict_to_df(ui_valid)
        ui_train.columns=['userid','itemid']
        ui_test.columns=['userid','itemid']
        ui_valid.columns=['userid','itemid']
        
        ui_train.to_csv(os.path.join(self.data_root,'train.csv'),index=None)
        ui_test.to_csv(os.path.join(self.data_root,'test.csv'),index=None)
        ui_valid.to_csv(os.path.join(self.data_root,'valid.csv'),index=None)
    
    def dict_to_df(self,dict):
        
        ks,vs=[],[]
        for k,v in dict.items():
            for i in v:
                ks.append(k)
                vs.append(i)
        df=pd.DataFrame()
        df[0]=ks
        df[1]=vs
        return df
    def load_data_df(self):
        
        data_tr=pd.read_csv(os.path.join(self.data_root,'train.csv'))
        data_te=pd.read_csv(os.path.join(self.data_root,'test.csv'))
        data_va=pd.read_csv(os.path.join(self.data_root,'valid.csv'))
        
        return data_tr,data_te,data_va

    def load_data(self, dataframe_type = None):
        
        data_tr=pd.read_csv(os.path.join(self.data_root,'train.csv'))
        data_te=pd.read_csv(os.path.join(self.data_root,'test.csv'))
        data_va=pd.read_csv(os.path.join(self.data_root,'valid.csv'))
        if dataframe_type:
            return data_tr, data_te, data_va
        
        data_tr=torch.tensor(data_tr.values)
        data_te=torch.tensor(data_te.values)
        if self.data_split_rate[1]!=0:
            data_va=torch.tensor(data_va.values)
        else:
            data_va=''
        return data_tr,data_te,data_va
        
    def get_user_items(self,userid):
        
        data=self.data_df
        
        user_column=torch.tensor(data['userid'].values)
        item_column=torch.tensor(data['itemid'].values)
        
        values=torch.ones(item_column.shape[0])
        Graph_index=torch.concat([user_column.reshape(1,-1),item_column.reshape(1,-1)],dim=0)

        sparse_graph=torch.sparse.FloatTensor(Graph_index
                                                    ,values
                                                    ,(self.user_num,self.item_num))
        return sparse_graph[userid]._indices()
    
    def get_item_users(self,itemid):
        
        data=self.data_df
        
        user_column=torch.tensor(data['userid'].values)
        item_column=torch.tensor(data['itemid'].values)
        
        values=torch.ones(item_column.shape[0])
        Graph_index=torch.concat([item_column.reshape(1,-1),user_column.reshape(1,-1)],dim=0)

        sparse_graph=torch.sparse.FloatTensor(Graph_index
                                                    ,values
                                                    ,(self.item_num,self.user_num))
        return sparse_graph[itemid]._indices().view(-1)

    def get_dataloader(self, batch_size=64, num_workers=0):
        
        tr_dataset,te_dataset,va_dataset=self.get_dataset()
        tr_loader=DataLoader(tr_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        te_loader=DataLoader(te_dataset,batch_size=batch_size,num_workers=num_workers)
        va_loader=DataLoader(va_dataset,batch_size=batch_size,num_workers=num_workers)
        return tr_loader,te_loader,va_loader
    
    def get_dataset(self):
        
        data_tr,data_te,data_va=self.load_data()
        train=dataset_bpr(data_tr,self.item_num,self.user_num,num_ng=5,is_training=True)
        test=dataset_bpr(data_te,self.item_num,self.user_num,num_ng=0)
        valid=dataset_bpr(data_va,self.item_num,self.user_num,num_ng=0)
        return train,test,valid

class dataset_universal(Dataset):
    def __init__(self,data,model_name=None,dataset_use=None):
        
        super().__init__()
        self.data=data
        self.model_name=model_name
        self.dataset_use=dataset_use
    def __len__(self):
        if self.model_name=='cp':
            if self.dataset_use=='train':
                return self.data.item_pop_train_num
            if self.dataset_use=='valid':
                return len(self.data.data_valid)
            if self.dataset_use=='test':
                return len(self.data.data_test)
    def __getitem__(self, index):
        if self.model_name=="cp":
            
            if self.dataset_use=='train':# item,user_vector
                item_index=self.data.item_population_train[index]
                return item_index,self.data.users_train_vector[index]
            if self.dataset_use=='valid':# user,item
                return self.data.data_valid[index,1],self.data.data_valid[index,0]
            if self.dataset_use=='test':# user,item
                
                users=self.data.data_test['userid'][index]
                items=self.data.data_test['itemid'][index]
                return users,items
class data_bpr(data_basic):
    def __init__(self,args):
        
        super().__init__(args,model_name='bpr',use_cols=[0,1])
        self.num_ng=args.num_ng
    
    def get_dataset(self):
        '''return train,test,valid
        '''
        data_tr,data_te,data_va=self.load_data()
        train=dataset_bpr(data_tr,self.item_num,self.user_num,num_ng=self.num_ng,is_training=True)
        test=dataset_bpr(data_te,self.item_num,self.user_num,num_ng=0)
        valid=dataset_bpr(data_va,self.item_num,self.user_num,num_ng=0)
        return train,test,valid
    
    def get_dataloader(self,batch_size=64,num_workers=0):
        tr_dataset,te_dataset,va_dataset=self.get_dataset()
        tr_loader=DataLoader(tr_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        te_loader=DataLoader(te_dataset,batch_size=batch_size,num_workers=num_workers)
        va_loader=DataLoader(va_dataset,batch_size=batch_size,num_workers=num_workers)
        return tr_loader,te_loader,va_loader
    
class data_cp():
    def __init__(self,args):
        
        self.data_split=args.split_rate
        self.data_name=args.data_name
        self.args=args
        self.data_root=os.path.join('data',args.data_name)
        c_print("Dataset: "+args.data_name)
        # load data
        self.data=pd.read_csv(os.path.join(self.data_root,'sm.csv'))      

        self.item_population=self.data['itemid'].value_counts().keys()
        self.item_population_num=len(self.item_population)
        
        self.train_rate=self.data_split[0]
        self.valid_rate,self.test_rate=self.data_split[1],self.data_split[2]
        self.load_data()
        
        self.item_population_train=self.data_train['itemid'].value_counts().keys()
        self.item_population_valid=self.data_valid['itemid'].value_counts().keys()
        self.item_population_test=self.data_test['itemid'].value_counts().keys()
        
        self.Info()
        
        self.users_train_vector=self.Get_Users_Vector(datause='train')
        self.users_test_vector=self.Get_Users_Vector(datause='test')

    def sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    
    def Get_Users_Vector(self,datause='data'):
        
        root_uv_train=os.path.join(self.data_root,'uv_train.npy')
        root_uv_valid=os.path.join(self.data_root,'uv_valid.npy')
        root_uv_test=os.path.join(self.data_root,'uv_test.npy')
        
        if datause=='train':
            items=self.item_population_train
        if datause=='valid':
            items=self.item_population_valid
        if datause=='test':
            items=self.item_population_test
        users_v=list()
        print('Geting user vector:')
        for i_idx in tqdm.tqdm(items,desc=datause+'set'):
            users_v.append(self.Get_User_Vector(i_idx,datause).reshape(1,-1))
        if datause=='train':
            users_vector_numpy=torch.concat(users_v,dim=0).numpy()
            # np.save(root_uv_train,users_vector_numpy)
        if datause=='valid':
            users_vector_numpy=torch.concat(users_v,dim=0).numpy()
            # np.save(root_uv_valid,users_vector_numpy)
        if datause=='test':
            users_vector_numpy=torch.concat(users_v,dim=0).numpy()
            # np.save(root_uv_test,users_vector_numpy)
        return torch.concat(users_v,dim=0)
    
    def Get_User_Vector(self,item_idx,datause):
        
        if datause=='data':
            data=self.data
        if datause=='train':
            data=self.data_train
        if datause=='valid':
            data=self.data_valid
        if datause=='test':
            data=self.data_test
            
        user_column=torch.tensor(data['userid'].values)
        item_column=torch.tensor(data['itemid'].values)
        
        values=torch.ones(item_column.shape[0])
        Graph_index=torch.concat([item_column.reshape(1,-1),user_column.reshape(1,-1)],dim=0)
        
        sparse_graph=torch.sparse.FloatTensor(Graph_index
                                                    ,values
                                                    ,(self.item_num,self.user_num))
        user_vecter=torch.zeros(self.user_num)
        user_vecter[sparse_graph[item_idx]._indices()]=1
        return user_vecter
    def get_user_items(self,user_idx):
        
        data=self.data
        user_column_idx=1
        
        user_column=torch.tensor(data['userid'].values)
        item_column=torch.tensor(data['itemid'].values)
        area_num=self.areas_num[user_column_idx-1]
        
        values=torch.ones(item_column.shape[0])
        Graph_index=torch.concat([user_column.reshape(1,-1),item_column.reshape(1,-1)],dim=0)
        
        sparse_graph=torch.sparse.FloatTensor(Graph_index
                                                    ,values
                                                    ,(area_num,self.item_num))
        
        return sparse_graph[user_idx]._indices()
    def get_item_users(self,itemid):
        
        data=self.data
        
        user_column=torch.tensor(data['userid'].values)
        item_column=torch.tensor(data['itemid'].values)
        
        values=torch.ones(item_column.shape[0])
        Graph_index=torch.concat([item_column.reshape(1,-1),user_column.reshape(1,-1)],dim=0)

        sparse_graph=torch.sparse.FloatTensor(Graph_index
                                                    ,values
                                                    ,(self.item_num,self.user_num))
        return sparse_graph[itemid]._indices().view(-1)
    def Get_super_matrix(self):
        
        return torch.tensor(self.data.values)

    def Get_Areas_num(self):
        area_nums=list()
        for col in self.data.columns[1:]:
            area_nums.append(len(self.data[col].value_counts().keys()))
        return area_nums
    def Info(self):
        self.data_len=len(self.data)
        self.data_train_len=len(self.data_train)
        self.data_valid_len=len(self.data_valid)
        self.data_test_len=len(self.data_test)
        self.item_pop_train_num=len(self.item_population_train)
        self.item_pop_valid_num=len(self.item_population_valid)
        self.item_pop_test_num=len(self.item_population_test)
        self.user_num=len(self.data['userid'].value_counts().keys())
        self.item_num=len(self.data['itemid'].value_counts().keys())
        self.aspect_num=len(self.data.columns)-1
        self.areas_num=self.Get_Areas_num()# list
        self.device='cpu'
        
        c_print("{}:\nInteraction num:{}\nUser num:{}\nItem num:{}".format(
            self.data_name,self.data_len,self.user_num,self.item_num))
        c_print("sparsity:{}".format(self.data_len/(self.user_num*self.item_num)))
    def Get_dataset(self,dataset_use=None):
        
        if dataset_use=='train':# 
            return dataset_universal(self,model_name='cp',dataset_use='train')
        if dataset_use=='test':# 
            return dataset_universal(self,model_name='cp',dataset_use='test')
        if dataset_use=='valid':# 
            return dataset_universal(self,model_name='cp',dataset_use='valid') 
    def get_dataloader(self,batch_size=64,dataset_use=None,train=True,num_workers=0):

        dt_tr=self.Get_dataset('train')
        dt_te=self.Get_dataset('test')
        dt_va=self.Get_dataset('valid')
        dl_tr=DataLoader(dt_tr,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        dl_te=DataLoader(dt_te,batch_size=batch_size,num_workers=num_workers)
        dl_va=DataLoader(dt_va,batch_size=batch_size,num_workers=num_workers)
        return dl_tr,dl_te,dl_va
    def load_data(self, dataframe_type = None):
                
        item_population=self.data['itemid'].value_counts().keys()
        item_user_dict=dict()# {item:users}
        
        self.data_train=pd.read_csv(os.path.join(self.data_root,'train.csv'))
        self.data_valid=pd.read_csv(os.path.join(self.data_root,'valid.csv'))
        self.data_test=pd.read_csv(os.path.join(self.data_root,'test.csv'))
        return self.data_train,self.data_test,self.data_valid

class dataset_bpr(Dataset):
    def __init__(self, features,
                 nuitem_num,num_user,num_ng=0,is_training=None):
        
        super(dataset_bpr, self).__init__()
        
        self.features = features
        self.nuitem_num = nuitem_num
        self.num_user = num_user
        self.num_ng = num_ng
        self.is_training = is_training
        if is_training:
            
            features = features.tolist()
            train_mat = sp.dok_matrix((num_user, nuitem_num), dtype=np.float32)
            for x in features:
                train_mat[x[0], x[1]] = 1.0
            self.train_mat = train_mat
            self.ng_sample()

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in tqdm.tqdm(self.features,desc='[ng_sampling]',leave=False):
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.nuitem_num)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.nuitem_num)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        if self.is_training:
            return user,item_i,item_j
        else:
            return user,item_i



def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


class data_lightgcn(data_basic):
    def __init__(self, args):
        super().__init__(args, model_name='lightgcn', use_cols=[0,1])
        self.num_ng=args.num_ng
        self.get_symmetrically_normalized_matrix()
    
    def get_dataloader(self, batch_size=64, num_workers=0):
        
        tr_dataset,te_dataset,va_dataset=self.get_dataset()
        tr_loader=DataLoader(tr_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        te_loader=DataLoader(te_dataset,batch_size=batch_size,num_workers=num_workers)
        va_loader=DataLoader(va_dataset,batch_size=batch_size,num_workers=num_workers)
        return tr_loader,te_loader,va_loader
    
    def get_dataset(self):
        
        data_tr,data_te,data_va=self.load_data()
        train=dataset_bpr(data_tr,self.item_num,self.user_num,num_ng=self.num_ng,is_training=True)
        test=dataset_bpr(data_te,self.item_num,self.user_num,num_ng=0)
        valid=dataset_bpr(data_va,self.item_num,self.user_num,num_ng=0)
        return train,test,valid
    
    def sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def get_symmetrically_normalized_matrix(self):
        
        save_path=os.path.join(self.data_root,
                               'symmetrically_normalized_adj_mat.npz')
        if os.path.exists(save_path):
            pre_adj_mat = sp.load_npz(save_path)
            # print("successfully loaded...")
            norm_adj = pre_adj_mat

            self.Graph = self.sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

            return self.Graph
        else:
            users=torch.tensor(self.data_df['userid']).view(1,-1)
            items=torch.tensor(self.data_df['itemid']).view(1,-1)
            adjacency_matrix=torch.concat([users,items],dim=0)

            # sub graph value construct
            values=torch.ones(adjacency_matrix.shape[1])
            # sparse graph construct
            # Rating matrix
            R=sp.csr_matrix((values,adjacency_matrix)
                                        ,(self.user_num,self.item_num))
            adj_mat = sp.dok_matrix((self.user_num + self.item_num, 
                                    self.user_num + self.item_num)
                                    , dtype=np.float32)
            # scipy.sparse has the following types of sp_matrix:csr, csc, lil, dok
            adj_mat = adj_mat.tolil()
            # training user item net
            R = R.tolil()        
            
            adj_mat[:self.user_num, self.user_num:] = R
            adj_mat[self.user_num:, :self.user_num] = R.T
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            print(save_path)
            sp.save_npz(save_path, norm_adj)

            self.Graph = self.sp_mat_to_sp_tensor(norm_adj)
            
            self.Graph = self.Graph.coalesce().to(self.device)

            return self.Graph

def dense_mat_to_coo_mat(mat):
    idx = torch.nonzero(mat).T
    data = mat[idx[0],idx[1]]
    coo_a = torch.sparse_coo_tensor(idx, data, mat.shape)
    return coo_a    

data_dict={'cp':data_cp,
           'bpr':data_bpr,
           'lightgcn':data_lightgcn}

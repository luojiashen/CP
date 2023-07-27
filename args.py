import argparse

class args:
    def __init__(self,model_name='cp'):
        self.data_name='citeulike'
        self.epoches=500 
        self.lr=0.0005
        self.device='cuda:0'
        self.batch_size=1024
        self.num_workers=0
        self.data_preprocess=0
        self.split_rate=[0.7,0.1,0.2]
        self.embedding_dim=64
        self.round_time=1
        self.model_args_init(model_name)
    def model_args_init(self,model_name):
        if model_name=='bpr':
            self.num_ng=5
        if model_name=='lightgcn':
            self.layer_num=3
            self.L2_coefficient=1e-5
            self.num_ng=5
        if model_name=='cp':
            self.layer_num = 3
            self.L2_lambda=1e-5
            data_hid={'lastfm':2,'citeulike':0,'frappe':0}
            self.hidden_num=data_hid[self.data_name]
            data_ind={'lastfm':5,'citeulike':5,'frappe':5.0}
            self.personality=data_ind[self.data_name]
            self.base_model='lightgcn'
            self.fit_loss=1


    def model_args_reset(self,model_name):
        if model_name=='cp':
            data_hid={'lastfm':2,'citeulike':0,'citeulike-t':0}
            self.hidden_num=data_hid[self.data_name]
            data_ind={'lastfm':5,'citeulike':13,'citeulike-t':7.5}
            self.personality=data_ind[self.data_name]

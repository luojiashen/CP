########################################################
import os
import time
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tqdm

import argparse
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import torch
from cprint import c_print
from record import Progress_Record,dict_to_text
from dataset import data_dict
from model import model_dict,Positive_Fit_Loss
from train import train_dict
import args as arguments
from evaluation import Recommender_evaluate
def Show_parameters(model):
    '''展示参数信息'''
    for p in model.named_parameters():
        print('Parameter name:{:25},{}, memory:{}M,require grad:{},Device:{}'.format(
                p[0],p[1].shape,p[1].nelement()/1e6,p[1].requires_grad,p[1].device))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
def experiment(model_name,args,a):
    # roots
    root='results'
    txt_root=os.path.join(root,f"{model_name}-{args.data_name}-{a}.txt")
    if not os.path.exists(root):
        os.makedirs(root)
    dict_to_text(txt_root,args.__dict__)

    best_record=Progress_Record(txt_root)
    for round in range(args.round_time):
        # data load
        data=data_dict[model_name]
        # model
        model=model_dict[model_name]
        # model train
        train=train_dict[model_name]
        data=data(args)
        model=model(data,args)

        # Show_parameters(model)
        round_best=train(model,data,args,txt_root)
        best_record.append_dict(round_best)
    best_record.mean()

def get_args():
    parser = argparse.ArgumentParser(description="experiment")

    # parser.add_argument('--args', type=str,default=[],nargs='+',
    #                     help="negtive sampling number")
    
    parser.add_argument('--models', type=str,default=['lightgcn','cp'],nargs='+',
                        help="models name")
    parser.add_argument('--data_name', type=str,default=[],nargs='+',
                        help="data name")
    parser.add_argument('--round_time', type=int,default=[],nargs='+',
                        help="independent repeated experiment times")
    parser.add_argument('--embedding_dim', type=int,default=[],nargs='+',
                        help="dimension of embedding")
    parser.add_argument('--epoches', type=int,default=[],nargs='+',
                        help="epoches")
    parser.add_argument('--batch_size', type=int,default=[],nargs='+',
                        help="batch size")
    parser.add_argument('--lr', type=float,default=[],nargs='+',
                        help="learning rate")
    
    parser.add_argument('--personality', type=float,default=[],nargs='+',
                        help="personality of CP RS")
    parser.add_argument('--hidden_num', type=int,default=[],nargs='+',
                        help="hidden num of UPN")
    parser.add_argument('--base_model', type=str,default=[],nargs='+',
                        help="consensus model of cp")
    parser.add_argument('--fit_loss', type=int,default=[],nargs='+',
                        help="use positive fit loss or not")
    parser.add_argument('--layer_num', type=int,default=[],nargs='+')
    return parser.parse_args()

if __name__=="__main__":

    experiment_args=get_args()
    c_print(f'experiment_args:{experiment_args}')

    for model_name in experiment_args.models:
        # args of model
        args=arguments.args(model_name)
        # args describe
        args_desc=''
        # re-inital args and do the model experiment
        # args initialize
        freeze_args=[]
        for arg in experiment_args.__dict__.keys():
            test_arg=experiment_args.__dict__[arg]
            if len(test_arg)==1:
                args.__dict__[arg]=test_arg[0]
                if arg=='data_name':
                    args.model_args_reset(model_name)
                if arg !='data_name' and arg != 'models':
                    args_desc+=str(arg)+'-'+str(test_arg[0])+'+'
                print(args_desc)
        # mutli-args
        sign=0
        for arg in experiment_args.__dict__.keys():
            if arg=='models':
                continue
            test_args=experiment_args.__dict__[arg]
            if len(test_args)>=2:# 指定了的模型参数组
                sign+=1
                for a in test_args:
                    args.__dict__[arg]=a
                    if arg!='data_name' and arg != 'models':
                        args.__dict__[arg]=a
                        c_print(args.__dict__)
                        experiment(model_name,args,args_desc+str(arg)+'-'+str(a))
                    else:
                        args.__dict__[arg]=a
                        if arg=='data_name':
                            args.model_args_reset(model_name)
                        c_print(args.__dict__)
                        experiment(model_name,args,args_desc[:-1])

        if sign==0:
            c_print(args.__dict__)
            experiment(model_name,args,args_desc[:-1])

            

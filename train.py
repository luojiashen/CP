from model import Positive_Fit_Loss
import tqdm
import torch
from dataset import data_dict
from model import model_dict,Positive_Fit_Loss
from evaluation import Recommender_evaluate
from record import Progress_Record
import os
from model import *
from cprint import c_print
import args as arguments

def train_bpr(model,data,args,root=None):
    save_path=os.path.join('show','bpr',args.data_name,'train')
    root_model=os.path.join('models',args.data_name)
    model_name='bpr'
    model_path = os.path.join(root_model,model_name+'.pt')
    model.to(device=args.device)

    record=Progress_Record(root)
    train_loader,test_loader,va=data.get_dataloader(batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epoches):
        model.train() 
        for user,item_i,item_j in tqdm.tqdm(train_loader,desc=f'epoch {epoch}',leave=False):
            user = user.to(device=args.device)
            item_i = item_i.to(device=args.device)
            item_j = item_j.to(device=args.device)
            model.zero_grad()
            prediction_i,prediction_j = model(user,item_i,item_j) 
            loss = -(prediction_i-prediction_j).sigmoid().log().sum() 
            loss.backward() 
            optimizer.step() 
        # test
        if epoch%5==0:
            eval_metrices=Recommender_evaluate(model=model,
                            data=data,
                            device=args.device)
            print(eval_metrices)
            stop=record.compare(eval_metrices)
            if stop:
                print("Not imporve for a long time")
                record.last()
                break
            elif stop==0:# performence imporved so model save
                if not os.path.exists(root_model):
                    os.makedirs(root_model)
                # add results to record
                record.append_dict(eval_metrices)
                if model.embs_dim != 2:
                    print(f'model save path:{model_path}')
                    torch.save(model,model_path)
                    record.last(path=os.path.join(root_model,model_name+'.txt'))

    return record.best()
def train_lightgcn(model,data,args,root=None):
    save_path=os.path.join('show','lightgcn',args.data_name,'train')
    root_model=os.path.join('models',args.data_name)
    model_name='lightgcn'
    model_path = os.path.join(root_model,model_name+'.pt')
    model.to(device=args.device)
    record=Progress_Record(root)
    record.count=2

    train_loader,test_loader,va=data.get_dataloader(batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    model.graph=data.get_symmetrically_normalized_matrix()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epoches):
        model.train() 
        for user,item_i,item_j in tqdm.tqdm(train_loader,desc=f'epoch {epoch}',leave=False):
            user = user.to(device=args.device)
            item_i = item_i.to(device=args.device)
            item_j = item_j.to(device=args.device)
            optimizer.zero_grad()
            bpr_loss,regularization_loss = model.loss(user,item_i,item_j) 
            loss=bpr_loss+model.L2_coefficient*regularization_loss
            loss.backward() 
            optimizer.step() 
        if epoch%20==0:
            eval_metrices=Recommender_evaluate(model=model,
                            data=data,
                            device=args.device)
            print(eval_metrices)
            stop=record.compare(eval_metrices)
            if stop:
                print("Not imporve for a long time")
                record.last()
                break
            elif stop==0:# performence imporved so model save
                if not os.path.exists(root_model):
                    os.makedirs(root_model)
                # add results to record
                record.append_dict(eval_metrices)
                if model.embs_dim != 2:
                    print(f'model save path:{model_path}')
                    torch.save(model,model_path)
                    record.last(path=os.path.join(root_model,model_name+'.txt'))# 保存当前模型的性能指标
        
        return record.best()


def train_cp(model:CP,data,args,root):
    base_model_path=os.path.join('models',args.data_name,args.base_model+'.pt')
    root_model=os.path.join('models',args.data_name)
    model_name='upp'
    model_path = os.path.join(root_model,model_name+'.pt')
    save_path=os.path.join('show','upp',args.data_name,'train')

    record=Progress_Record(root)
    base_model=torch.load(os.path.join('models',args.data_name,args.base_model+'.pt'))
    dl_tr,dl_te,_=data.get_dataloader(batch_size=args.batch_size)
    if args.fit_loss==1:
        loss=Positive_Fit_Loss(individuality=args.personality)
    else:
        loss=torch.nn.CrossEntropyLoss()
    model.item_embs=base_model.item_embs.weight.data

    optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    for epoch in range(args.epoches):
        lr=list()
        model.to(device=args.device)
        loss.to(device=args.device)
        model.train()
        for item_idx,user_vec in tqdm.tqdm(dl_tr,leave=False):
            item_idx=item_idx.to(device=args.device)
            user_vec=user_vec.to(device=args.device)
            y_pred=model(item_idx)
            l=loss(y_pred,user_vec)
            optim.zero_grad()
            l.backward()
            optim.step()
            lr.append(l.item())
            
        if epoch%5==0:
            eval_metrices=Recommender_evaluate(model=model,
                            data=data,
                            device=args.device)
            print(eval_metrices)
            stop=record.compare(eval_metrices)
            if stop:
                print("Not imporve for a long time")
                record.last()
                break
            elif stop==0:# performence imporved so model save
                if not os.path.exists(root_model):
                    os.makedirs(root_model)
                # add results to record
                record.append_dict(eval_metrices)
                if model.embs_dim != 2:
                    print(f'model save path:{model_path}')
                    torch.save(model,model_path)
                    record.last(path=os.path.join(root_model,model_name+'.txt'))# 
    return record.best()
   
train_dict={'cp':train_cp,
            'bpr':train_bpr,
            'lightgcn':train_lightgcn}


##################evaluation indicator###########
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import roc_auc_score,log_loss
import torch
import tqdm
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import warnings
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

warnings.filterwarnings("ignore")

def Recall_MLC(y_true,y_pred,recall_n=None):
    
    if recall_n:
        preds, indices=torch.topk(y_pred,k=recall_n)
        y_pred=torch.zeros_like(y_pred)
        for i,pos in enumerate(indices):
            y_pred[i][pos]=1# 将每个batch排名前n的物品赋值为1
    # average='samples'表示为求所有实例的均值
    return recall_score(y_true=y_true,y_pred=y_pred,average='samples')
def Recall(y_true,y_pred,recall_n):
    
    return torch.tensor(recall_score(y_true=y_true,y_pred=y_pred))


def Hit(y_true_idx, rec_list, N):
    top_N = rec_list.squeeze()[:N]
    hit_N = 1.0 if y_true_idx in top_N else 0.0
    return hit_N

def HR(y_true,y_pred,hr_n):
    if torch.sum(y_true)!=0:
        rec_list=torch.argsort(y_pred,dim=0,descending=True)# 降序
        y_true_list=y_true.nonzero().squeeze().reshape(-1)
        hit=0
        for true_idx in y_true_list:
            hit+=Hit(true_idx,rec_list,hr_n)
        return hit/len(y_true_list)
    
def HR_MLC(y_true,y_pred,hr_n):
    hr=0
    count=0# hr有效计数
    for y_t,y_p in zip(y_true,y_pred):
        if HR(y_t,y_p,hr_n):
            count+=1
            hr+=HR(y_t,y_p,hr_n)
    if not count:
        return 0
    else:
        return hr/count
def Evaluation_metrics(y_true,y_pred):
    return {
            'recall@10':Recall_MLC(y_true,y_pred,10),
            'recall@20':Recall_MLC(y_true,y_pred,20),
            'recall@30':Recall_MLC(y_true,y_pred,30),
            'hit@10':HR_MLC(y_true,y_pred,10),
            }
# def Eval_Non_NegSample(y_true,y_pred,hr_n=5):
#     return {
#             'hr_n':HR(y_true,y_pred,hr_n)}
# def get_implict_matrix(rec_items, test_set):
#     rel_matrix = [[0] * rec_items.shape[1] for _ in range(rec_items.shape[0])]
#     for user in range(len(test_set)):
#         for index, item in enumerate(rec_items[user]):
#             if item in test_set[user]:
#                 rel_matrix[user][index] = 1
#     return np.array(rel_matrix)


# def Label_to_test_set(y_true):
#         test_set=list()
#         for y in y_true:
#             test_set.append(y.nonzero().view(-1).tolist())
#         return test_set

def Recommender_evaluate(model, data, device='cpu'):
    
    test_loader=data.get_dataloader()[1]
    train_df = data.load_data(dataframe_type = True)[0]
    umax,imax=data.user_num, data.item_num

    y_true=torch.zeros(size=(umax,imax)).to(device=device)# shape:[user num,item num]
    y_pred=torch.zeros(size=(umax,imax)).to(device=device)
    model.eval()
    model.to(device=device)
    
    for u,i in test_loader:
        u=u.to(device=device)
        i=i.to(device=device)
        y_true[u,i]=1
    y_true=y_true.cpu()
    
    with torch.no_grad():
        for u in tqdm.tqdm(range(umax),desc='Evaluation'):
            u_t=torch.tensor(u,dtype=torch.long).view(-1).to(device=device)
            l=imax# item num
            i=torch.range(0,l-1
                        ,dtype=torch.long).to(device=device)
            y_pred[u_t,i]=model.predict(u_t,i).view(-1)
    # drop train data
    for uid,iid in tqdm.tqdm(zip(train_df['userid'],train_df['itemid'])):
        y_pred[uid,iid]=0
    evaluate_metrics=Evaluation_metrics(y_true, y_pred.cpu().detach())
    return evaluate_metrics


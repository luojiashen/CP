import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import yaml
class Progress_Record:
    def __init__(self,save_path=None):
        
        self.record_dict=dict()
        self.best_record=dict()
        self.stop_count = 15
        self.count=0# 
        self.path=save_path
    def append(self,key,value):
        if key not in self.record_dict.keys():
            self.record_dict[key]=list()
        self.record_dict[key].append(value)
    def append_record(self,record):
        for k,v in record.record_dict.items():
            self.record_dict[k]=v
    def append_dict(self,d,datause=''):
        for k,v in d.items():
            self.append(k+datause,v)
        pass
    def print(self):
        f=''
        for k,v in self.record_dict.items():
            f+=f'{k}:{v[-1]:.6f}\t'
        print(f)
    def get_records(self,key):
        return self.record_dict[key]
    def __len__(self):
        if len(self.record_dict)==0:
            return 0
        else:
            for k in self.record_dict.keys():
                return len(self.record_dict[k])
    def last(self,path=None):
        if path:
            save_path=path
            with open(save_path,mode='w') as f:
                f.write('')
        elif self.path:
            save_path=self.path
        for k,v in self.record_dict.items():
            output=f'{k} in last epoch:\t{v[-1]}\n'
            if path:
                with open(save_path,mode='a') as f:
                    f.write(output)
            elif self.path:
                print(output)
                with open(save_path,mode='a') as f:
                    f.write(output)

    def plot(self,keys=None):
        plt.close()
        if not keys:
            keys=self.record_dict.keys()
        for k in keys:
            plt.plot(self.get_records(k),label=k)
        plt.grid()
        plt.legend()
        plt.savefig(self.path)

    def mean(self):
        if self.path:
            save_path=self.path
        for k,v in self.record_dict.items():
            output=f'{k} mean(round {len(v)}):\t{sum(v)/len(v)}\n'
            print(output)
            if self.path:
                print("Mean save path:",save_path)
                with open(save_path,mode='a') as f:
                    f.write(output)

    def compare(self,result):
        '''early stop
        '''
        sign=0
        for k,v in result.items():
            if k not in self.best_record.keys():
                self.best_record[k]=v
            else:
                if self.best_record[k]<v:
                    self.best_record[k]=v
                else:
                    sign+=1
        if sign==len(result):
            self.count+=1
        else:
            print('performence imporved')
            self.count=0
            return 0 
        if self.count>=self.stop_count:
            return 1
        return None
    def save(self):
        with open(self.path+'.yaml','w') as f:
            yaml.dump(self.record_dict,stream=f, allow_unicode=True) 

    def best(self):
        best_dict=dict()
        for k in self.record_dict.keys():
            if 'logloss' in k:
                min_k=min(self.record_dict[k])
                best_dict[k]=min_k
                best_describe="best {} is:{:.4f} in epoch:{}".format(k,min_k,self.record_dict[k].index(min_k))
                print(best_describe)
            else:
                max_k=max(self.record_dict[k])
                best_dict[k]=max_k
                best_describe="best {} is:{:.4f} in epoch:{}".format(k,max_k,self.record_dict[k].index(max_k))
                print(best_describe)
        return best_dict
    

def dict_to_text(path,dict):
    with open(path, mode="w") as f:
        json.dump(dict, f, indent=4)
        
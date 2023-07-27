## TITLE: CP

## Dataset

  We use three datasets of our paper, lastfm, citeulike-t and citeulike, the links are shown as follow:  [http://www.lastfm.com](http://www.lastfm.com/) , http://www.citeulike.org/faq/data.adp 

## Environment

```
python==3.7.0
torch==1.10.1
numpy==1.20.3
scipy==1.7.3
sklearn==0.0
pandas==1.3.4
matplotlib==3.4.3
```



## Training and Pre-Training

If model folder alright have the pre-trained LightGCN model(which we alright have), you could run the next command directly.

```
python main.py --models cp --data_name lastfm
```

If model folder does not have the pre-trained model, you should pre-train the base model before you train the cp based model. For example if you want run the cp-lightgcn, the command is: 

```
python main.py --models lightgcn cp --data_name lastfm
python main.py --models lightgcn cp --data_name citeulike
python main.py --models lightgcn cp --data_name citeuslike-t
```

If you want to train CP based model in different dataset continuously, the command is:

```
python main.py --models cp --data_name lastfm frappe_interact citeulike
```


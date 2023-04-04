
import torch
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit
import numpy as np

def split_geometric(dataset, K = 5, train_size = 0.75, random_seed=42):
    X = np.zeros((len(dataset), dataset[0].x.shape[1]))
    label = torch.tensor([])
    for data in (dataset):
        if len(data.y.shape)==1:
            label = torch.cat((label, data.y),0)
        elif len(data.y.shape)>=2 and data.y.shape[1]>1:
            label = torch.cat((label, data.y[:,1]),0)
        else:
            label = torch.cat((label, data.y[:,0]),0)
    if K>=2:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_seed)
        return (skf.split(X=X, y=label.numpy(force=True)))
    elif K==1:
        sss = StratifiedShuffleSplit(n_splits=K, random_state= random_seed, train_size=train_size)
        return sss.split(X=X, y=label.numpy(force=True))

def split_regr(dataset, K = 5, train_size = 0.75, random_seed=42):
    ssp = ShuffleSplit(n_splits=K, train_size=train_size, random_state=random_seed)
    X = np.zeros((len(dataset)))
    return ssp.split(X)
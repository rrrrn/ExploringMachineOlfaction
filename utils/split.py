from sklearn.model_selection import (
    ShuffleSplit,
)
import numpy as np


def split_regr(dataset, K=5, train_size=0.75, random_seed=42):
    ssp = ShuffleSplit(n_splits=K, train_size=train_size, random_state=random_seed)
    X = np.zeros((len(dataset)))
    return ssp.split(X)

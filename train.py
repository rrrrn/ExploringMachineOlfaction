from torch_geometric.loader import DataLoader
import torch
from utils.train_one_round import train, train_regr, test_regr
from utils.split import split_regr
from datamodule.datasets.kellerdataset import KellerDataset
from datamodule.datasets.dravnieks_dataset import DravnieksDataset
import torch.nn.functional as F
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from models.nn.GENConv import GCN2Regressor

if __name__ == "__main__":
    num_epoch = 2000
    K = 5
    n_epochs_stop = 100
    datasetname = "dravnieks"  ## either dravnieks or keller
    path = f"results/{datasetname}"

    if datasetname == "dravnieks":
        dataset = DravnieksDataset(mode="cv")  # specify dataset of interest
    elif datasetname == "keller":
        dataset = KellerDataset(mode="cv")
    else:
        raise ValueError

    criterion = (
        torch.nn.HuberLoss()
    )  # less sensitive to outliers in data than L2-norm MSE

    pool_dim = 175  # Dimension of the pooling layer
    hidden_channels = [15, 20, 27, 36]  # dimensions of each GNN layers
    fully_connected = [96, 63]  # dimension of fully-connected neural net
    model = GCN2Regressor(
        node_feature_dim=dataset.num_node_features,
        edge_feature_dim=dataset.num_edge_features,
        hidden_channels=hidden_channels,
        pool_dim=pool_dim,
        fully_connected_channels=[96, 63],
        output_channels=dataset.num_classes,
    )

    seed_everything(432)

    ## split the entire dataset into training and testing.
    split_tt = split_regr(dataset=dataset, K=1)
    for i, (train_idx, val_idx) in enumerate(split_tt):
        trainset = dataset[train_idx]
        testset = dataset[val_idx]
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    ## split the test for train-validation (5-fold)
    split = split_regr(dataset=trainset)
    trainr2, trainmse, testr2, testmse = [], [], [], []
    for i, (train_idx, val_idx) in enumerate(split):
        ## reset param for model
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        min_val_loss = np.inf
        train_dataset = trainset[train_idx]
        val_dataset = trainset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=18, shuffle=True)
        lc_valloss, lc_trainloss = [], []
        refloss, epochs_no_improve = 0, 0
        for epoch in tqdm(range(1, num_epoch)):
            loss, error = train_regr(model, train_loader, criterion=criterion)
            val_loss, val_error = test_regr(model, val_loader, criterion)
            lc_valloss += [abs(val_loss)]
            lc_trainloss += [abs(loss.numpy(force=True))]

            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1

            if epoch > 5 and epochs_no_improve == n_epochs_stop:
                early_stop = True
                break

        _, error = test_regr(model, test_loader, criterion=criterion)
        error = pd.DataFrame(error)

        expvar = str(error["explained_variance_weighted"]).split(".")[1][:4]
        ## save the well-performed models
        if error["explained_variance_weighted"].unique() > 0.130:
            print("saved!")
            (pd.DataFrame(error)).to_csv(f"{path}/gnn_regr/{expvar}.csv")
            torch.save(model, f"{path}/gnn_regr/{expvar}.pt")

from torch_geometric.loader import DataLoader
import torch
from data.datasets.kellerdataset import KellerDataset
from data.datasets.dravnieks_dataset import DravnieksDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.split import split_regr
from utils.train_one_round import train_transfer, test_regr
from models.nn.GENConv import GCN2Regressor
from pytorch_lightning.utilities.seed import seed_everything
import os
from tqdm import tqdm

if __name__ == "__main__":
    criterion = torch.nn.HuberLoss()
    dataset = DravnieksDataset(
        mode="all"
    )  ## the dataset that the model originally is trained on
    dataset_target = KellerDataset(
        mode="all", test=True
    )  ## specifies the transfer learning target, exclude overlapped molecules
    num_epoch = 1000
    batch_size = 16
    n_epochs_stop = 100
    K = 5
    min_val_loss = np.inf
    path = f"results/dravnieks/gnn_regr/"
    modeldir = os.listdir(path=path)

    seed_everything(432)
    model = GCN2Regressor(
        node_feature_dim=dataset.num_node_features,
        edge_feature_dim=dataset.num_edge_features,
        hidden_channels=[15, 20, 27, 36],
        pool_dim=175,
        fully_connected_channels=[96, 63],
        output_channels=dataset.num_classes,
    )
    modelname = "1413"
    model = torch.load(path + modelname + ".pt")

    ## freeze pre-trained parameters
    for name, param in model.named_parameters():
        if "regressor" in name:
            pass
        else:
            param.requires_grad = False

    ## transfer learning for regression
    last_layer_dim = model.regressor[8].in_features
    model.regressor[8] = torch.nn.Linear(last_layer_dim, dataset_target.num_classes)

    ## collect the last-layer param to update
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    ## split train and test set
    split = split_regr(K=1, dataset=dataset_target)
    for i, (train_index, test_index) in enumerate(split):
        train_data = dataset_target[train_index]
        test_data = dataset_target[test_index]

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    split_train = split_regr(K=K, dataset=train_data)

    ## k-fold training and testing
    for i, (train_ind, test_ind) in enumerate(split_train):
        train_loader = DataLoader(
            dataset_target[train_ind], batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            dataset_target[test_ind], batch_size=batch_size, shuffle=False
        )

        for epoch in tqdm(range(1, num_epoch)):
            train_transfer(
                model=model,
                loader=train_loader,
                criterion=criterion,
                params=params_to_update,
            )
            val_loss, val_error = test_regr(model, val_loader, criterion)

            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1

            if epoch > 5 and epochs_no_improve == n_epochs_stop:
                early_stop = True
                train_loss, train_error = test_regr(
                    model, test_loader, criterion=criterion
                )
                break
            elif epoch + 1 == num_epoch:
                train_loss, train_error = test_regr(
                    model, test_loader, criterion=criterion
                )
            else:
                continue

        test_loss, test_error = test_regr(model, test_loader, criterion=criterion)
        pd.DataFrame(test_error).to_csv(
            f"results/transfer_ml/gnn/{modelname}_fold{i}.csv"
        )

from torch_geometric.loader import DataLoader
import torch
from data.datasets.dravnieks_dataset import DravnieksDataset
from data.datasets.kellerdataset import KellerDataset
import pandas as pd
from utils.train_one_round import callmodel
from models.nn.GENConv import GCN2Regressor


def learn_embeddings(datasetname, model="GCN2Regressor"):
    """[aims to compute and save GNN-learned embeddings for each dataset.
    Computed embeddings are used as feature input to Machine Learning (ML) algorithms,
    as compared to Mordred descriptors]

    Arguments:
        datasetname {string} -- [on which dataset the embedding is extracted] "keller" or "dravnieks"
        model {string/model object} -- [the GNN model used for produce embeddings]

    Returns:
        None
    """

    path = f"results/{datasetname}/gnn_regr"  # the path to load original models
    if datasetname == "keller":
        gnnname = "1440"
        dataset = KellerDataset(mode="all")
    else:
        gnnname = "1413"
        dataset = DravnieksDataset(mode="all")

    ## create a dataloader using dataset-of-interest
    target_loader = DataLoader(dataset=dataset, batch_size=18, shuffle=False)

    ## initiate the GCN model for regression
    if model == "GCN2Regressor":
        model = GCN2Regressor(
            node_feature_dim=dataset.num_node_features,
            edge_feature_dim=dataset.num_edge_features,
            hidden_channels=[15, 20, 27, 36],
            pool_dim=175,
            fully_connected_channels=[108, 56, 12],
            output_channels=dataset.num_classes,
        )
        model = torch.load(f"{path}/{gnnname}.pt")
    else:
        raise ValueError

    ## call the model and extract learned embeddings as well as predicting target
    features, y = callmodel(target_loader, model=model)
    features = features.numpy(force=True)
    y = y.numpy(force=True)
    pd.DataFrame(features).to_csv(f"{path}/{gnnname}_embeddings.csv")
    pd.DataFrame(y).to_csv(f"{path}/{gnnname}_target.csv")

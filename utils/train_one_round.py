import torch
from torchmetrics.functional.classification import binary_auroc, binary_accuracy
from torchmetrics.functional import (
    explained_variance,
    mean_squared_error,
    pearson_corrcoef,
)
import numpy as np


def train(model, loader, criterion):
    """
    train deep learning 'model' for binary classification and return important metrics
    loss 'criterion' and data 'loader' depends on the usage scenario
    metrics of fit focus on *binary-auroc* and *accuracy*, regardless of loss criterion choice
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    model.train()
    for i, data in enumerate(
        loader, 0
    ):  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # Perform a single forward pass
        loss = criterion(out[:, 1].float(), data.y.float())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        y_truth = torch.tensor([])
        if len(y_truth) == 0:
            out_proba = out
            y_truth = data.y
        else:
            y_truth = torch.cat([y_truth, data.y])
            out_proba = torch.vstack([out_proba, out])
        _, pred = torch.max(out_proba, 1)
    scheduler.step()

    return (
        loss,
        binary_accuracy(pred, y_truth).numpy(force=True),
        binary_auroc(out_proba[:, 1], y_truth).numpy(force=True),
    )


def train_regr(model, loader, criterion):
    """
    train deep learning 'model' for regression and return important metrics
    loss 'criterion' and data 'loader' depends on the usage scenario
    metrics of fit focus on *explained_variance*, *correlation*, and *mean-squared-error*, regardless of loss criterion choice
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    model.train()
    for i, data in enumerate(
        loader, 0
    ):  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # Perform a single forward pass.]
        loss = criterion(out.float(), data.y.float())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        y_truth = torch.tensor([])
        if len(y_truth) == 0:
            out_all = out
            y_truth = data.y
        else:
            y_truth = torch.cat([y_truth, data.y])
            out_all = torch.vstack([out_all, out])
    scheduler.step()

    ## compute mesure of fit during training phase
    error = dict()
    loss = criterion(out_all, y_truth)
    error["loss"] = loss.numpy(force=True)
    error["explained_variance"] = explained_variance(
        out_all, y_truth, multioutput="raw_values"
    ).numpy(force=True)
    error["explained_variance_weighted"] = explained_variance(
        out_all, y_truth, multioutput="variance_weighted"
    ).numpy(force=True)
    error["corr"] = pearson_corrcoef(out_all, y_truth).numpy(force=True)
    error["mse"] = mean_squared_error(out_all, y_truth).numpy(force=True)

    return loss, error


def train_transfer(model, loader, criterion, params):
    """
    similar to <train_regr> function but train deep learning 'model' on specified 'params' instead of all
    loss 'criterion' and data 'loader' depends on the usage scenario
    no loss or metrics of fit will be reported
    """
    optimizer = torch.optim.Adam(params=params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    model.train()
    for i, data in enumerate(
        loader, 0
    ):  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out.float(), data.y.float())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    scheduler.step()


def test(model, loader, criterion):
    """
    compute errors and loss on data 'loader' in model evaluation mode
    metrics of interests include *accuracy* and *binary-auroc*
    """
    model.eval()
    pred = torch.tensor([])
    y_truth = torch.tensor([])
    out_proba = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(
            loader, 0
        ):  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if len(y_truth) == 0:
                out_proba = out
                y_truth = data.y
            else:
                y_truth = torch.cat([y_truth, data.y])
                out_proba = torch.vstack([out_proba, out])
        _, pred = torch.max(out_proba, 1)

    return (
        binary_accuracy(pred, y_truth).item(),
        binary_auroc(out_proba[:, 1], y_truth).item(),
        criterion(out_proba[:, 1].float(), y_truth.float()).item(),
    )


def test_regr(model, loader, criterion):
    """
    compute errors and loss on data 'loader' in model evaluation mode
    metrics of interests include *explained_variance*, *correlation*, *mean_squared_error*
    """
    model.eval()
    y_truth = torch.tensor([])
    out_all = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(
            loader, 0
        ):  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if len(out_all) == 0:
                out_all = out
                y_truth = data.y
            else:
                y_truth = torch.cat((y_truth, data.y), 0)
                out_all = torch.cat((out_all, out), 0)

    ## compute model performance on test data
    error = dict()
    mse = np.zeros(out_all.shape[1])
    if out_all.shape[1] != 1:
        for i in range(out_all.shape[1]):
            mse[i] = mean_squared_error(out_all[:, i], y_truth[:, i]).numpy(force=True)
    loss = criterion(out_all, y_truth)

    error["loss"] = loss.numpy(force=True)
    error["explained_variance"] = explained_variance(
        out_all, y_truth, multioutput="raw_values"
    ).numpy(force=True)
    error["explained_variance_weighted"] = explained_variance(
        out_all, y_truth, multioutput="variance_weighted"
    ).numpy(force=True)
    error["corr"] = pearson_corrcoef(out_all, y_truth).numpy(force=True)
    error["mse"] = mse

    return loss, error


def get_features(name, features):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


def callmodel(loader, model, embeddings=True):
    """
    extract embeddings at the pooling layer by registering forward hook
    corresponding target values are collected and returned for easier acccess
    """
    if embeddings:
        feature_all = []
        features = {}
        model.pool.register_forward_hook(get_features("embeddings", features=features))
    else:
        feature_all = None

    model.eval()
    ## prepare to return corresponding target values
    y = []

    for i, data in enumerate(
        loader, 0
    ):  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        if embeddings:
            if len(feature_all) == 0:
                feature_all = features["embeddings"]
            else:
                feature_all = torch.concat((feature_all, features["embeddings"]), 0)

        if len(y) == 0:
            y = data.y
        else:
            y = torch.concat((y, data.y), 0)

    return feature_all, y

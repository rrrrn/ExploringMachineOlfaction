import pandas as pd
import numpy as np
import logging
import yaml
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
from tqdm import tqdm
from joblib import dump

warnings.filterwarnings("ignore")


def prepare_data(dataset="dravnieks", numpy_form=True, test=False):
    """[preprocess data to fit the model input format]

    Arguments:
        dataset {string} -- name of the dataset we are using: dravnieks or keller
        numpy_form {bool} -- if False, then output the original pandas dataframe
        test {bool} -- if True, then exclude all overlapped chemicals

    Returns:
        [dict] -- {data set, target}
    """
    data = pd.read_csv(f"data/{dataset.lower()}/raw/descriptors.csv").iloc[:, 1:]
    target = pd.read_csv(f"data/{dataset.lower()}/raw/{dataset.lower()}_pa.csv").iloc[
        :, 1:
    ]

    if dataset.lower() == "dravnieks":
        target.drop(columns=["IsomericSMILES", "IUPACName"], inplace=True)
    elif dataset.lower() == "keller":
        target["CID"] = data["CID"].values
    else:
        raise ValueError

    target.dropna(inplace=True)

    if test:
        ovlp = np.load("data/overlapped.npy")
        for cid in ovlp:
            data = data[data["CID"] != cid]
            target = target[target["CID"] != cid]

    X = data.fillna(0)

    if "CID" in X.columns:
        X.drop(columns="CID", inplace=True)
    if "CID" in target.columns:
        target.drop(columns="CID", inplace=True)

    if numpy_form:
        X = X.to_numpy()
        Y = target.to_numpy()
    else:
        Y = target

    return {"data": X, "target": Y}


def log(path, file):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    """
    This script carries out hyperparameter search for classical machine learning methods supported by sklearn library

    Model selection criteria: explained-variance, a finite-version of coefficient of determinant (r2_score)

    Models are cross-validated on each dataset odor descriptor-wise,
    Best-model, best-parameters, and cross-validation metrics are saved to individual files.
    """

    ## model type
    modelsets = [
        LinearRegression(),
        SVR(),
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
    ]
    datasetname = "keller"  # specify the dataset, either "dravnieks" or "keller"
    metricname = ["explained_variance", "neg_mean_squared_error"]
    randomseed = 432
    cvsplit = KFold(n_splits=5, shuffle=True, random_state=randomseed)
    for ml_model in modelsets:
        modelname = str(ml_model)[: len(str(ml_model)) - 2]

        path = f"results/{datasetname}/"

        ## prepare data for grid search
        data = prepare_data(datasetname, numpy_form=False)
        X = data["data"]
        descriptor = np.load("data/retained_descriptors.npy", allow_pickle=True)
        X = X[descriptor]
        y = data["target"]
        col = y.columns

        # X_train, y_train, X_test, y_test = train_test_split(X, y, train_size =.75, random_state = randomseed)

        logger = log(path="logs/", file=modelname.lower() + ".logs")
        logger.info("-" * 15 + "Start Session!" + "-" * 15)

        # load grid parameters
        with open("configs/param_search/" + modelname.lower() + ".yaml", "r") as stream:
            parameters = yaml.safe_load(stream)

        if not os.path.isdir(f"{path}/best_models/{modelname}"):
            os.makedirs(f"{path}/best_models/{modelname}")
        if not os.path.isdir(f"{path}/best_params/"):
            os.makedirs(f"{path}/best_params/")
        if not os.path.isdir(f"{path}/metrics/"):
            os.makedirs(f"{path}/metrics/")

        logger.info("{} regressor parameter grid search".format(modelname))

        # start grid search
        bestscore, best_param = dict(), dict()
        for i in tqdm(range(len(y.columns))):
            descriptor_name = col[i]
            if "/" in descriptor_name:
                descriptor_name = descriptor_name.replace("/", "_")

            bestscore[descriptor_name] = np.zeros(2)
            grid_search = GridSearchCV(
                ml_model,
                parameters,
                cv=cvsplit,
                scoring=(metricname),
                refit="explained_variance",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X, y.iloc[:, i])
            results = grid_search.cv_results_

            for i, scorer in enumerate(metricname):
                best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
                bestscore[descriptor_name][i] = results["mean_test_%s" % scorer][
                    best_index
                ]

            best_param[descriptor_name] = list(grid_search.best_params_.values())

            ## save the best trained model object
            dump(
                grid_search.best_estimator_,
                f"{path}/best_models/{modelname}/{descriptor_name}.joblib",
            )

        best_param = (pd.DataFrame(best_param, index=list(parameters.keys()))).T
        best_param.to_csv(f"{path}/best_params/{modelname}_param.csv")
        best_score = pd.DataFrame(bestscore, index=metricname).T
        best_score.to_csv(f"{path}/metrics/{modelname}.csv")

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
    data= pd.read_csv(f"data/{dataset.lower()}/raw/imputed_descriptor.csv").iloc[:,1:]

    if dataset.lower()=="dravnieks":
        target = pd.read_csv(f"data/{dataset.lower()}/raw/{dataset.lower()}_new.csv").iloc[:,1:]
        target.drop(columns=['IsomericSMILES', 'IUPACName'], inplace=True)
    elif dataset.lower()=="keller":
        target = pd.read_csv(f"data/{dataset.lower()}/raw/{dataset.lower()}_pa.csv").iloc[:,1:]
        target["CID"]=data["CID"].values
    else:
        raise ValueError

    target.dropna(inplace=True)

    if test:
        ovlp = np.load("data/overlapped.npy")
        for cid in ovlp:
            data = data[data["CID"]!=cid]
            target = target[target["CID"]!=cid]

    X = data.fillna(0)

    if "CID" in X.columns:
        X.drop(columns="CID", inplace=True)
    if "CID" in target.columns:
        target.drop(columns="CID", inplace=True)

    if numpy_form:
        X = X.to_numpy()
        Y = target.to_numpy()
    else:
        Y =target
    
    return {"data":X, "target":Y}


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


if __name__=='__main__':
    '''
    This script carries out hyperparameter search for classical machine learning methods supported by sklearn library
    
    Model selection criteria: explained-variance, a finite-version of coefficient of determinant (r2_score)
    
    Models are cross-validated on each dataset odor descriptor-wise,
    Best-model, best-parameters, and cross-validation metrics are saved to individual files.
    '''

    ## model type
    ml_model = KNeighborsRegressor()
    randomseed = 432
    model_name=str(ml_model)[:len(str(ml_model))-2]
    datasetname = "dravnieks" # specify the dataset, either "dravnieks" or "keller"
    metricname = "explained_variance"

    path = f"results/transfer_ml/embeddings_to_{datasetname}"

    ## prepare data for grid search
    data = prepare_data(datasetname,numpy_form=False)
    X = data["data"]
    descriptor = np.load("data/retained_descriptors.npy", allow_pickle = True)
    X = X[descriptor]
    y = data["target"]
    col = y.columns

    # X_train, y_train, X_test, y_test = train_test_split(X, y, train_size =.75, random_state = randomseed)

    logger = log(path="logs/", file=model_name.lower()+".logs")
    logger.info("-"*15+"Start Session!"+"-"*15)

    # load grid parameters
    with open("configs/param_search/"+model_name.lower()+".yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    
    if not os.path.isdir(f"{path}/best_models/{model_name}"):
        os.makedirs(f"{path}/best_models/")
    if not os.path.isdir(f"{path}/best_params/"):
        os.makedirs(f"{path}/best_params/")
    if not os.path.isdir(f"{path}/{metricname}/"):
        os.makedirs(f"{path}/{metricname}/")

    logger.info("{} regressor parameter grid search".format(model_name))

    # start grid search
    best_score, best_param = dict(), dict()
    for i in tqdm(range(len(y.columns))):
        descriptor_name = col[i]
        if "/" in descriptor_name:
            descriptor_name = (descriptor_name.replace("/","_"))
        
        grid_search = GridSearchCV(ml_model, parameters, 
        cv=KFold(n_splits=5, shuffle=True, random_state=432), scoring=(metricname))
        grid_search.fit(X, y.iloc[:,i])
        best_score[descriptor_name] = grid_search.best_score_
        best_param[descriptor_name] = list(grid_search.best_params_.values())

        ## save the best trained model object
        dump(grid_search.best_estimator_, f"{path}/best_models/{model_name}/{descriptor_name}.joblib")
    
    best_param = (pd.DataFrame(best_param, index=list(parameters.keys()))).T
    best_param.to_csv(f"{path}/best_params/{model_name}_param.csv")
    best_score = pd.DataFrame(best_score, index=range(1)).T
    best_score.to_csv(f"{path}/{metricname}/{model_name}_score.csv")

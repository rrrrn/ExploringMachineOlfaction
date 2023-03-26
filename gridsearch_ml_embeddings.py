import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
from tqdm import tqdm
from joblib import dump, load
from gridsearch_ml_mordred import *
from embeddings import learn_embeddings


if __name__=='__main__':
    '''
    This script carries out grid-search for ML models on GNN-learned embeddings specific to each dataset
    GNN-learned embeddings are learn
    '''
    # model type
    ml_model = KNeighborsRegressor()
    randomseed = 432
    datasetname = "dravnieks"
    metricname = "explained_variance"

    model_name=str(ml_model)[:len(str(ml_model))-2]
    path = f"results/{datasetname}/gnn_regr"
    if not os.path.isfile(f"results/transfer_ml/embeddings_to_{datasetname}.csv"):
        learn_embeddings(datasetname=datasetname)
    targetpath = f"results/transfer_ml/embeddings_to_{datasetname}.csv"

    if datasetname=="keller":
        gnnmodel = "1440"
    elif datasetname=="dravnieks":
        gnnmodel = "1413"

    X = pd.read_csv(f"{path}/{gnnmodel}_embeddings.csv").iloc[:,1:]
    y = pd.read_csv(f"{path}/{gnnmodel}_target.csv").iloc[:,1:]
    data = prepare_data(datasetname,numpy_form=False)
    col = data["target"].columns
    
    assert y.shape[1]==len(col)

    logger = log(path="logs/", file=model_name.lower()+".logs")
    logger.info("-"*15+"Start Session!"+"-"*15)

    # load grid parameters
    with open("configs/param_search/"+model_name.lower()+".yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    
    if not os.path.isdir(f"{path}/best_models/{model_name}"):
        os.makedirs(f"{path}/best_models/{model_name}")
    if not os.path.isdir(f"{path}/best_params/{model_name}"):
        os.makedirs(f"{path}/best_params/{model_name}")

    logger.info("{} regressor parameter grid search".format(model_name))

    # start grid search
    best_score,best_param = dict(), dict()
    for i in tqdm(range(len(y.columns))):
        descriptor_name = col[i]
        if "/" in descriptor_name:
            descriptor_name = (descriptor_name.replace("/","_"))
        
        grid_search = GridSearchCV(ml_model, parameters, 
        cv=KFold(n_splits=5, shuffle=True, random_state=432), scoring=(metricname))
        grid_search.fit(X, y.iloc[:,i])
        best_score[descriptor_name] = grid_search.best_score_
        best_param[descriptor_name] = list(grid_search.best_params_.values())

        dump(grid_search.best_estimator_, f"{path}/best_models/{model_name}/{descriptor_name}.joblib")
    
    best_param = (pd.DataFrame(best_param, index=list(parameters.keys()))).T
    best_param.to_csv(f"{targetpath}/best_params/{model_name}_param.csv")
    best_score = pd.DataFrame(best_score, index=range(1)).T
    best_score.to_csv(f"{targetpath}/{metricname}/{model_name}_score.csv")

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    explained_variance_score,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr
from joblib import load

from gridsearch_ml_mordred import prepare_data


if __name__ == "__main__":
    path = "results/dravnieks/best_models"
    train_wrap = prepare_data(dataset="Dravnieks", numpy_form=False)
    test_wrap = prepare_data(dataset="Keller", numpy_form=False, test=True)

    # descriptors

    X_test = test_wrap["data"]
    y_test = test_wrap["target"]

    labeldravnieks = [
        "SWEET",
        "GARLIC, ONION",
        "SWEATY",
        "PUTRID, FOUL, DECAYED",
        "HERBAL, GREEN,CUTGRASS",
    ]
    labelkeller = ["SWEET ", "GARLIC ", "SWEATY ", "DECAYED", "GRASS "]

    modelname = "GradientBoostingRegressor"

    ## load models
    mse_error = np.zeros(len(labelkeller))
    explained_var = np.zeros(len(labelkeller))
    mape = np.zeros(len(labelkeller))
    corr = np.zeros(len(labelkeller))
    for i in range(len(labelkeller)):
        model = load(f"{path}/{modelname}/{labeldravnieks[i]}.joblib")
        descriptor = model.feature_names_in_
        preds = model.predict(X_test[descriptor])
        mse_error[i] = mean_squared_error(y_pred=preds, y_true=y_test[labelkeller[i]])
        explained_var[i] = explained_variance_score(
            y_pred=preds, y_true=y_test[labelkeller[i]]
        )
        mape[i] = mean_absolute_percentage_error(
            y_pred=preds, y_true=y_test[labelkeller[i]]
        )
        corr[i] = pearsonr(preds, y_test[labelkeller[i]])[0]

    dferror = pd.DataFrame()
    dferror["mse"] = np.array(mse_error)
    dferror["corr"] = np.array(corr)
    dferror["mape"] = np.array(mape)
    dferror["explained_variance"] = np.array(explained_var)
    dferror["drav_label"] = labeldravnieks
    dferror["keller_label"] = labelkeller

    dferror.to_csv(f"results/transfer_ml/mordred/{modelname}.csv")

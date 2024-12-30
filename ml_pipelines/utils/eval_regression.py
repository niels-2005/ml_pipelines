import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, save_folder: str = "save_folder"
):
    df_dict = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "rmsle": np.log(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }
    df = pd.DataFrame(df_dict).to_csv(f"{save_folder}/model_metrics.csv")

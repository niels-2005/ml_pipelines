import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, save_folder: str = "save_folder"
) -> None:
    """
    Calculates common regression metrics and saves the results as a CSV file.

    The function computes key regression metrics such as Mean Absolute Error (MAE),
    Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Root Mean Squared Logarithmic
    Error (RMSLE), and the R-squared (R²) score. The metrics are saved in a CSV file
    named "model_metrics.csv" in the specified folder.

    Args:
        y_true (np.ndarray): Ground truth (true) target values, with shape (n_samples,).
        y_pred (np.ndarray): Predicted target values, with shape (n_samples,).
        save_folder (str): Directory where the metrics CSV file will be saved. Default is "save_folder".

    Returns:
        None: The function saves the metrics to a CSV file and does not return any value.
    """
    df_dict = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "rmsle": np.log(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }
    pd.DataFrame([df_dict]).to_csv(f"{save_folder}/model_metrics.csv", index=False)

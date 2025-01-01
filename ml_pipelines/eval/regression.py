import os

from ml_pipelines.utils import calculate_regression_metrics


class RegressionEvaluator:
    """
    A utility class for evaluating regression models. It provides functionality to
    compute and save various regression metrics to assess model performance.

    Args:
        y_true (list or array-like): Ground truth values.
        y_pred (list or array-like): Predicted values.
        save_folder (str): Directory to save evaluation results.

    Attributes:
        y_true (list or array-like): Ground truth values.
        y_pred (list or array-like): Predicted values.
        save_folder (str): Directory where evaluation results are saved.
    """

    def __init__(self, y_true, y_pred, save_folder):
        self.y_true = y_true
        self.y_pred = y_pred

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            self.save_folder = save_folder
        else:
            self.save_folder = save_folder

    def start_evaluation(self):
        """
        Calculates and saves key regression metrics for evaluating the model's performance.

        The results are saved in the specified folder as a CSV file.
        """
        calculate_regression_metrics(self.y_true, self.y_pred, self.save_folder)

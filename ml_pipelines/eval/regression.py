import os

from ml_pipelines.utils import calculate_regression_metrics


class RegressionEvaluator:
    def __init__(self, y_true, y_pred, save_folder):
        self.y_true = y_true
        self.y_pred = y_pred

        if not os.path.exists(save_folder):
            self.save_folder = save_folder
        else:
            self.save_folder = save_folder

    def start_evaluation(self):
        calculate_regression_metrics(self.y_true, self.y_pred, self.save_folder)

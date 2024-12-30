import os
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
# import scikitplot as skplt
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)

from ml_pipelines.utils.eval_classification import (get_wrong_predictions,
                                                    make_classification_report,
                                                    make_confusion_matrix, make_precision_recall_curve, make_roc_curve)


class BaseClassificationEvaluator:
    def __init__(
        self,
        y_true,
        y_pred,
        class_names: list,
        y_probas=None,
        average_metrics: str = "weighted",
        save_folder: str = "model_evaluation",
        cm_figsize: tuple[int, int] = (10, 10),
        report_figsize: tuple[int, int] = (12, 6),
        curve_figsize: tuple[int, int] = (12, 6)
    ):
        """
        A utility class for evaluating classification models. It provides
        functionality to compute and save confusion matrices, classification reports,
        and various evaluation metrics. Additionally, it can generate ROC and
        precision-recall curves if probability scores are provided.

        Args:
            y_true (list or array-like): Ground truth labels.
            y_pred (list or array-like): Predicted labels.
            class_names (list): Names of the classes corresponding to labels.
            y_probas (array-like, optional): Predicted probabilities for each class. Defaults to None.
            average_metrics (str, optional): Averaging method for metrics ('weighted', 'macro', etc.). Defaults to "weighted".
            save_folder (str, optional): Directory to save evaluation results. Defaults to "model_evaluation".
            cm_figsize (tuple[int, int], optional): Size of the confusion matrix plot. Defaults to (10, 10).
            report_figsize (tuple[int, int], optional): Size of the classification report plot. Defaults to (12, 6).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_probas = y_probas
        self.class_names = class_names
        self.average = average_metrics
        self.cm_figsize = cm_figsize
        self.report_figsize = report_figsize
        self.curve_figsize = curve_figsize

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            self.save_folder = save_folder
        else:
            self.save_folder = save_folder

    def create_confusion_matrix(self):
        """
        Generates and saves the confusion matrix plot for the model's predictions.
        """
        make_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.save_folder,
            self.class_names,
            self.cm_figsize,
        )

    def create_classification_report(self):
        """
        Generates and saves a classification report (in CSV format) and a
        visualization of the report.
        """
        report = classification_report(
            self.y_true, self.y_pred, target_names=self.class_names, output_dict=True
        )
        report_df = pd.DataFrame(report).to_csv(
            f"{self.save_folder}/classification_report.csv"
        )
        make_classification_report(report, self.save_folder, self.report_figsize)

    def calculate_metrics(self):
        """
        Calculates accuracy, F1-score, precision, and recall based on the provided
        averaging method. Saves the metrics in a CSV file.
        """
        acc_score = accuracy_score(self.y_true, self.y_pred) * 100
        f1 = f1_score(self.y_true, self.y_pred, average=self.average) * 100
        precision = (
            precision_score(self.y_true, self.y_pred, average=self.average) * 100
        )
        recall = recall_score(self.y_true, self.y_pred, average=self.average) * 100

        df_dict = {
            f"accuracy": [round(acc_score, 2)],
            f"f1-score_{self.average}": [round(f1, 2)],
            f"precision_{self.average}": [round(precision, 2)],
            f"recall_{self.average}": [round(recall, 2)],
        }
        pd.DataFrame(df_dict).to_csv(
            f"{self.save_folder}/model_metrics_{self.average}.csv"
        )

    def create_roc_curve(self):
        """
        Generates and saves the ROC curve plot. Requires `y_probas` to be defined.

        Raises:
            ValueError: If `y_probas` is not provided.
        """
        if self.y_probas is None:
            raise ValueError(f"y_probas is not defined.")
        make_roc_curve(self.y_true, self.y_probas, self.class_names, self.save_folder, self.curve_figsize)

    def create_precision_recall_curve(self):
        """
        Generates and saves the precision-recall curve plot. Requires `y_probas`
        to be defined.

        Raises:
            ValueError: If `y_probas` is not provided.
        """
        if self.y_probas is None:
            raise ValueError(f"y_probas is not defined.")
        make_precision_recall_curve(self.y_true, self.y_probas, self.class_names, self.save_folder, self.curve_figsize)


class NumClassificationEvaluator(BaseClassificationEvaluator):
    def __init__(
        self,
        y_true,
        y_pred,
        class_names: list,
        y_probas=None,
        average_metrics: str = "weighted",
        save_folder: str = "model_evaluation",
        cm_figsize: tuple[int, int] = (10, 10),
        report_figsize: tuple[int, int] = (12, 6),
        curve_figsize: tuple[int, int] = (12, 6)
    ):
        """
        A utility class for evaluating numerical classification models. It provides
        functionality to compute and save confusion matrices, classification reports,
        and various evaluation metrics. Additionally, it can generate ROC and
        precision-recall curves if probability scores are provided.

        Args:
            y_true (list or array-like): Ground truth labels.
            y_pred (list or array-like): Predicted labels.
            class_names (list): Names of the classes corresponding to labels.
            y_probas (array-like, optional): Predicted probabilities for each class. Defaults to None.
            average_metrics (str, optional): Averaging method for metrics ('weighted', 'macro', etc.). Defaults to "weighted".
            save_folder (str, optional): Directory to save evaluation results. Defaults to "model_evaluation".
            cm_figsize (tuple[int, int], optional): Size of the confusion matrix plot. Defaults to (10, 10).
            report_figsize (tuple[int, int], optional): Size of the classification report plot. Defaults to (12, 6).

        Example usage:
            evaluator = NumClassificationEvaluator(
                y_true=[0, 1, 0, 1],
                y_pred=[0, 1, 1, 1],
                class_names=["Class 0", "Class 1"],
                y_probas=[[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]],
            )
            evaluator.start_evaluation()
        """
        super().__init__(
            y_true,
            y_pred,
            class_names,
            y_probas,
            average_metrics,
            save_folder,
            cm_figsize,
            report_figsize,
            curve_figsize
        )
        self.y_probas = y_probas

    def start_evaluation(self):
        """
        Runs the full evaluation pipeline, including confusion matrix creation,
        metrics calculation, and optional ROC and precision-recall curves.
        """
        super().create_confusion_matrix()
        super().create_classification_report()
        super().calculate_metrics()

        if self.y_probas is not None:
            super().create_roc_curve()
            super().create_precision_recall_curve()


class TextClassificationEvaluator(BaseClassificationEvaluator):
    def __init__(
        self,
        X_test,
        y_true,
        y_pred,
        class_names: list,
        task: Literal["binary", "multiclass"],
        y_probas=None,
        average_metrics: str = "weighted",
        save_folder: str = "model_evaluation",
        cm_figsize: tuple[int, int] = (10, 10),
        report_figsize: tuple[int, int] = (12, 6),
        predictions_figsize: tuple[int, int] = (12, 6)
    ):
        """
        A utility class for evaluating text classification models. It provides
        functionality to compute and save confusion matrices, classification reports,
        and various evaluation metrics. Additionally, it can generate ROC and
        precision-recall curves if probability scores are provided.

        Args:
            y_true (list or array-like): Ground truth labels.
            y_pred (list or array-like): Predicted labels.
            class_names (list): Names of the classes corresponding to labels.
            y_probas (array-like, optional): Predicted probabilities for each class. Defaults to None.
            average_metrics (str, optional): Averaging method for metrics ('weighted', 'macro', etc.). Defaults to "weighted".
            save_folder (str, optional): Directory to save evaluation results. Defaults to "model_evaluation".
            cm_figsize (tuple[int, int], optional): Size of the confusion matrix plot. Defaults to (10, 10).
            report_figsize (tuple[int, int], optional): Size of the classification report plot. Defaults to (12, 6).

        Example usage:
            evaluator = NumClassificationEvaluator(
                y_true=[0, 1, 0, 1],
                y_pred=[0, 1, 1, 1],
                class_names=["Class 0", "Class 1"],
                y_probas=[[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]],
            )
            evaluator.start_evaluation()
        """
        super().__init__(
            y_true,
            y_pred,
            class_names,
            y_probas,
            average_metrics,
            save_folder,
            cm_figsize,
            report_figsize,
        )
        self.X_test = X_test
        self.y_probas = y_probas
        self.task = task 
        self.predictions_figsize = predictions_figsize


    def start_evaluation(self):
        """
        Runs the full evaluation pipeline, including confusion matrix creation,
        metrics calculation, and optional ROC and precision-recall curves.
        """
        super().create_confusion_matrix()
        super().create_classification_report()
        super().calculate_metrics()
        self.get_predictions()

        if self.y_probas is not None:
            super().create_roc_curve()
            super().create_precision_recall_curve() 

    def get_predictions(self):
        get_wrong_predictions(
            self.X_test, 
            self.y_true,
            self.y_pred, 
            self.class_names, 
            self.task, 
            self.save_folder, 
            self.predictions_figsize
        )


    

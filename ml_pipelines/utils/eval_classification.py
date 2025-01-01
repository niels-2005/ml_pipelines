import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.preprocessing import label_binarize


def make_classification_report(
    report: dict, save_folder: str, figsize: tuple[int, int]
) -> None:
    """
    Generates and saves a visual representation of a classification report.

    Args:
        report (dict): The classification report as a dictionary, typically obtained
                       from `sklearn.metrics.classification_report` with `output_dict=True`.
        save_folder (str): Path to the directory where the plot will be saved.
        figsize (tuple[int, int]): Size of the output figure (width, height).

    Returns:
        None: The function saves the classification report as an image in the specified folder.
    """
    labels = list(report.keys())[:-3]
    metrics = ["precision", "recall", "f1-score", "support"]
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(data, cmap="coolwarm")
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.title("Classification Report with Support")
    plt.savefig(f"{save_folder}/classification_report.png")


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_folder: str,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
) -> None:
    """
    Generates and saves a confusion matrix plot comparing ground truth labels with predictions.

    Args:
        y_true (np.ndarray): Array of ground truth labels.
        y_pred (np.ndarray): Array of predicted labels.
        save_folder (str): Path to the directory where the plot will be saved.
        classes (np.ndarray, optional): Array of class names corresponding to label indices.
                                         Defaults to None, in which case indices are used.
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (10, 10).
        text_size (int, optional): Font size for the labels and annotations. Defaults to 15.
        cmap (str, optional): Colormap for the confusion matrix. Defaults to "Blues".
        norm (bool, optional): Whether to normalize values. Defaults to False.

    Returns:
        None: The function saves the confusion matrix as an image in the specified folder.
    """
    cm = (
        confusion_matrix(y_true, y_pred, normalize="true")
        if norm
        else confusion_matrix(y_true, y_pred)
    )
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)
    labels = classes if classes is not None else np.arange(len(cm))
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )
    plt.tight_layout()
    plt.savefig(f"{save_folder}/confusion_matrix.png")


def calculate_num_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, save_folder: str, average: str = "weighted"
) -> None:
    """
    Calculates key numerical metrics for classification models and saves them as a CSV file.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        save_folder (str): Path to the directory where the metrics CSV will be saved.
        average (str, optional): Type of averaging for metrics (e.g., 'micro', 'macro', 'weighted').
                                 Defaults to "weighted".

    Returns:
        None: The metrics are saved as a CSV file in the specified folder.
    """
    acc_score = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average=average) * 100
    precision = precision_score(y_true, y_pred, average=average) * 100
    recall = recall_score(y_true, y_pred, average=average) * 100
    df_dict = {
        f"accuracy": [round(acc_score, 2)],
        f"f1-score_{average}": [round(f1, 2)],
        f"precision_{average}": [round(precision, 2)],
        f"recall_{average}": [round(recall, 2)],
    }
    pd.DataFrame(df_dict).to_csv(f"{save_folder}/model_metrics_{average}.csv")


def get_wrong_predictions(
    X_test: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    save_folder: str,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies and saves incorrect predictions made by a model, along with their details.

    Args:
        X_test (pd.Series): Test inputs corresponding to predictions.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        classes (list): List of class names corresponding to label indices.
        save_folder (str): Path to the directory where outputs will be saved.
        figsize (tuple[int, int], optional): Size of the plot for visualization. Defaults to (10, 6).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames:
            - All predictions with additional metadata.
            - Subset of incorrect predictions only.
    """
    y_pred = y_pred.reshape(-1) if len(classes) == 2 else y_pred
    y_true = y_true.reshape(-1) if len(classes) == 2 else y_true
    df_dict = {
        "text": X_test,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_classnames": [classes[i] for i in y_true],
        "y_pred_classnames": [classes[i] for i in y_pred],
    }
    df_pred = pd.DataFrame(df_dict).reset_index(drop=True)
    df_pred["pred_correct"] = df_pred["y_true"] == df_pred["y_pred"]
    df_pred.to_csv(f"{save_folder}/predictions.csv")
    plt.figure(figsize=figsize)
    sns.countplot(x="pred_correct", hue="y_pred_classnames", data=df_pred)
    plt.title("Balance between Predictions")
    plt.savefig(f"{save_folder}/predictions_balance.png")
    wrong_preds = df_pred[df_pred["pred_correct"] == False].reset_index(drop=True)
    wrong_preds.to_csv(f"{save_folder}/wrong_predictions.csv")


def make_precision_recall_curve(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    class_names: list[str],
    save_folder: str,
    figsize: tuple[int, int],
) -> None:
    """
    Plots a precision-recall curve for binary or multiclass classification.

    The function computes the precision-recall curve for each class (or a single curve for binary classification)
    and displays them in a single plot. The area under the curve (AUC) is also calculated and displayed in the legend.

    Args:
        y_true (np.ndarray): Ground truth labels, with shape (n_samples,).
        y_probas (np.ndarray): Predicted probabilities for each class, with shape (n_samples, n_classes).
        class_names (list[str]): List of class names (length must match the number of classes).
        save_folder (str): Folder where the plot will be saved.
        figsize (tuple[int, int]): Size of the plot as (width, height).

    Returns:
        None: The function saves the plot as "precision_recall_curve.png" in the specified folder.
    """
    if len(class_names) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_probas[:, 1])
        auc_score = average_precision_score(y_true, y_probas[:, 1])
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, lw=2, label=f"(area = {auc_score:.3f})")
    else:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        plt.figure(figsize=figsize)
        for i in range(len(class_names)):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_probas[:, i]
            )
            auc_score = average_precision_score(y_true_bin[:, i], y_probas[:, i])
            plt.plot(
                recall,
                precision,
                lw=2,
                label=f"{class_names[i]} (area = {auc_score:.3f})",
            )

    plt.title("Precision-Recall Curve", fontsize=16)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.grid(True)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/precision_recall_curve.png")


def make_roc_curve(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    class_names: list[str],
    save_folder: str,
    figsize: tuple[int, int],
) -> None:
    """
    Plots an ROC curve for binary or multiclass classification.

    The function computes the Receiver Operating Characteristic (ROC) curve for each class
    and displays them in a single plot. It calculates the area under the curve (AUC) for each class.

    Args:
        y_true (np.ndarray): Ground truth labels, with shape (n_samples,).
        y_probas (np.ndarray): Predicted probabilities for each class, with shape (n_samples, n_classes).
        class_names (list[str]): List of class names (length must match the number of classes).
        save_folder (str): Folder where the plot will be saved.
        figsize (tuple[int, int]): Size of the plot as (width, height).

    Returns:
        None: The function saves the plot as "roc_curve.png" in the specified folder.
    """
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probas[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, lw=2, label=f"(area = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2, label="No Skill")
    else:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        plt.figure(figsize=figsize)
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probas[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (area = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2, label="No Skill")

    plt.title("ROC Curve", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.grid(True)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/roc_curve.png")

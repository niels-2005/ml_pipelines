import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.preprocessing import label_binarize


def make_classification_report(report: dict, save_folder: str, figsize: tuple):
    labels = list(report.keys())[:-3]
    metrics = ["precision", "recall", "f1-score", "support"]
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(data, cmap="coolwarm")
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    # Adding the text
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
    Makes a labelled confusion matrix comparing predictions and ground truth labels, with options to normalize
    and save the figure.

    Args:
      y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
      y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
      classes (np.ndarray): Array of class labels (e.g., string form). If `None`, integer labels are used.
      figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
      text_size (int): Size of output figure text (default=15).
      norm (bool): If True, normalize the values in the confusion matrix (default=False).
      savefig (bool): If True, save the confusion matrix plot to the current working directory (default=False).

    Returns:
        None: This function does not return a value but displays a Confusion Matrix. Optionally, it saves the plot.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10,
                            norm=True,
                            savefig=True)
    """
    # Create the confusion matrix
    cm = (
        confusion_matrix(y_true, y_pred, normalize="true")
        if norm
        else confusion_matrix(y_true, y_pred)
    )

    # Plot the figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    # Set class labels
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(len(cm))

    # Set the labels and titles
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

    # Annotate the cells with the appropriate values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )

    plt.tight_layout()
    plt.savefig(f"{save_folder}/confusion_matrix.png")


def calculate_num_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, save_folder: str, average: str = "weighted"
):
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
    df_metrics = pd.DataFrame(df_dict).to_csv(
        f"{save_folder}/model_metrics_{average}.csv"
    )


def get_wrong_predictions(
    X_test: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    save_folder: str,
    figsize: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies and returns the correct and incorrect predictions made by a classification model.
    The function creates a DataFrame that includes the test inputs, actual and predicted labels, and class names.
    It also visualizes the distribution of correct and incorrect predictions.

    Args:
        X_test (pd.Series): The input text data that was used for testing the model, used here to trace back incorrect predictions to the original inputs.
        y_true (np.ndarray): The actual labels from the test data, representing the true classes of the inputs.
        y_pred (np.ndarray): The predicted labels produced by the classification model, used to compare against the true labels to determine prediction correctness.
        classes (list): A list of class names corresponding to the label indices, used to convert label indices into human-readable class names for easier interpretation and visualization.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. The first DataFrame includes all predictions with columns for the text, actual and predicted labels, and whether each prediction was correct.
            2. The second DataFrame is a subset of the first and includes only the rows where the predictions were incorrect.

    The function also plots a count plot showing the balance between correct and incorrect predictions across predicted class labels.
    """

    if len(classes) == 2:
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

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


def make_precision_recall_curve(y_true, y_probas, class_names, save_folder, figsize):
    """
    Plots a beautiful precision-recall curve for both binary and multiclass classification.
    It computes the precision-recall curve for each class and displays them all
    in a single plot.

    Args:
        y_true (array-like): Ground truth labels.
        y_probas (array-like): Predicted probabilities for each class.
        class_names (list): List of class names (length must match the number of classes).
        save_folder (str): Folder where the plot will be saved.
        figsize (tuple): Size of the plot.
    """
    if len(class_names) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_probas[:, 1])
        auc_score = average_precision_score(y_true, y_probas[:, 1])
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, lw=2, label=f" (area = {auc_score:.3f})")
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


def make_roc_curve(y_true, y_probas, class_names, save_folder, figsize):
    """
    Plots a beautiful ROC curve for both binary and multiclass classification.
    It computes the ROC curve for each class and displays them all
    in a single plot.

    Args:
        y_true (array-like): Ground truth labels.
        y_probas (array-like): Predicted probabilities for each class.
        class_names (list): List of class names (length must match the number of classes).
        save_folder (str): Folder where the plot will be saved.
        figsize (tuple): Size of the plot.
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

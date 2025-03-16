import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

### PLOT CLASS DISTRIBUTION BEFORE/AFTER SMOTE
def plot_class_distribution(y_train_before, y_train_after, save_path=None):
    """
    Plots the class distribution before and after SMOTE.
    
    Args:
        y_train_before (array-like): Labels before applying SMOTE.
        y_train_after (array-like): Labels after applying SMOTE.
        save_path (str, optional): Path to save the figure. If None, the plot is shown.
    """
    plt.figure(figsize=(12, 5))

    # Before SMOTE
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train_before, palette=["#0101DF", "#DF0101"])
    plt.title("Class Distribution BEFORE SMOTE")
    plt.xlabel("Class (0: No Fraud, 1: Fraud)")
    plt.ylabel("Count")

    # After SMOTE
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_train_after, palette=["#0101DF", "#DF0101"])
    plt.title("Class Distribution AFTER SMOTE")
    plt.xlabel("Class (0: No Fraud, 1: Fraud)")
    plt.ylabel("Count")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Class distribution plot saved at {save_path}")
    else:
        plt.show()


### PLOT TRAINING LOSS OVER ITERATIONS
def plot_training_loss(minibatch_loss_list, save_path=None):
    """
    Plots the training loss over iterations.
    
    Args:
        minibatch_loss_list (list): List of loss values during training.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(minibatch_loss_list)), minibatch_loss_list, label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Training loss plot saved at {save_path}")
    else:
        plt.show()


### PLOT ACCURACY, PRECISION, RECALL, F1-SCORE OVER EPOCHS
def plot_metrics(train_metrics_list, valid_metrics_list, save_path=None):
    """
    Plots training and validation accuracy, precision, recall, and F1-score.

    Args:
        train_metrics_list (list of dicts): Training metrics per epoch.
        valid_metrics_list (list of dicts): Validation metrics per epoch.
        save_path (str, optional): Path to save the figure.
    """
    epochs = len(train_metrics_list)

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    titles = ["Accuracy", "Precision", "Recall", "F1-Score"]

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(1, epochs + 1), [m[metric] for m in train_metrics_list], label="Train")
        plt.plot(range(1, epochs + 1), [m[metric] for m in valid_metrics_list], label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.title(f"{titles[i]} Over Epochs")
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Metrics plot saved at {save_path}")
    else:
        plt.show()


### PLOT ROC & PRECISION-RECALL CURVES
def plot_roc_pr_curves(y_true, y_probs, save_path=None):
    """
    Plots ROC and Precision-Recall curves.

    Args:
        y_true (array-like): True labels.
        y_probs (array-like): Predicted probabilities.
        save_path (str, optional): Path to save the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(recall, precision)

    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC-ROC: {auc_roc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")  # Random guess
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AUC-PR: {auc_pr:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f" ROC & PR curves saved at {save_path}")
    else:
        plt.show()

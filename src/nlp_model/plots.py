import torch as tc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import Tuple

def plot_metrics_vs_epochs(
    history,
    path: str, 
    dpi: int, 
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots the training and validation set loss.

    Parameters
    ----------
    history : 
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    epochs_mesh = tc.arange(0, len(train_loss))

    fig, ax = plt.subplots(figsize=figsize)

    plt.plot(epochs_mesh, train_loss, label="training")
    plt.plot(epochs_mesh, eval_loss, label="validation")

    ax.set_xlabel("epochs", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)
    ax.legend(fontsize=14)
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"loss_vs_epochs.png", bbox_inches="tight", dpi=dpi)

def compute_confusion_matrix(
    y_true, 
    y_pred
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot()

def plot_ROC_curve(
    y_true, 
    y_prob,
    path: str, 
    dpi: int, 
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots the training and validation set loss.

    Parameters
    ----------
    y_true : 
    y_prob : 
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.legend(fontsize=14)
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"loss_vs_epochs.png", bbox_inches="tight", dpi=dpi)
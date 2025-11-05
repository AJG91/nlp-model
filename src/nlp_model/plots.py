import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    path: str,
    dpi: int
) -> None:
    """
    Compute and save a confusion matrix plot for binary classification results.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_pred : np.ndarray
        Predicted class labels. Must have the same shape as `y_true`.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot()

    fig = disp.figure_        
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"confusion_matrix.png", bbox_inches="tight", dpi=dpi)
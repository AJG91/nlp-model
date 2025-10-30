from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def compute_confusion_matrix(
    y_true, 
    y_pred,
    path,
    dpi
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot()

    fig = disp.figure_        
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"confusion_matrix.png", bbox_inches="tight", dpi=dpi)
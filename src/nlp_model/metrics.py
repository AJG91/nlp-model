import torch as tc
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(
    eval_pred: tuple[tc.tensor, tc.tensor]
) -> dict[str, float]:
    """
    Compute evaluation metrics (accuracy and weighted F1 score) for model predictions.

    Parameters
    ----------
    eval_pred : tuple[tc.tensor, tc.tensor]
        A tuple containing predictions (output logits or probabilities) and labels (ground-truth) of the evaluation phase.

    Returns
    -------
    metrics : dict[str, float]
        A dictionary containing accuracy and F1 score.
    """
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(
    eval_pred
):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
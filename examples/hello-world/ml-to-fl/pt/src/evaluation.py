import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def compute_metrics(model, data_loader, device, apply_mask=False):
    """
    Compute accuracy, precision, recall, F1-score, and AUC-ROC for a fraud detection model.
    """
    model.eval()
    all_predictions, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            targets = targets.view(-1).cpu().numpy()

            logits = model(features, apply_mask=apply_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions = (probabilities >= 0.5).astype(int)  # Fix: Lowered threshold to 0.5

            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_probs.extend(probabilities)

    return {
        "accuracy": accuracy_score(all_targets, all_predictions) * 100,
        "precision": precision_score(all_targets, all_predictions, zero_division=0) * 100,
        "recall": recall_score(all_targets, all_predictions, zero_division=0) * 100,
        "f1_score": f1_score(all_targets, all_predictions, zero_division=0) * 100,
        "auc_roc": roc_auc_score(all_targets, all_probs) * 100,
        "auc_pr": average_precision_score(all_targets, all_probs) * 100
        
    }
def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset and print key metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device ('cpu' or 'cuda') to perform computations.
    """
    metrics = compute_metrics(model, test_loader, device)

    print("**Evaluation Results:**")
    print(f"Accuracy   : {metrics['accuracy']:.2f}%")
    print(f"Precision  : {metrics['precision']:.2f}%")
    print(f"Recall     : {metrics['recall']:.2f}%")
    print(f"F1-score   : {metrics['f1_score']:.2f}%")
    print(f"AUC-ROC    : {metrics['auc_roc']:.2f}%")  # New metric
    print(f"AUC-PR    : {metrics['auc_pr']:.2f}%")  # New metric

    return metrics

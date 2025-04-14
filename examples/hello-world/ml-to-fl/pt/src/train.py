import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from distutils.version import LooseVersion as Version
from nvflare.client.tracking import SummaryWriter
from evaluation import compute_metrics  # Call evaluation function only
from data import get_dataloaders_fraud  # Import dataset functions

def set_all_seeds(seed):
    """Set seed to ensure reproducibility."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic():
    """Enable deterministic mode to ensure stable results."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if torch.__version__ <= Version("1.7"):
        torch.set_deterministic(True)
    else:
        torch.use_deterministic_algorithms(True)

def train_model(model, num_epochs, train_loader, valid_loader, test_loader, 
                optimizer, criterion, device, input_model=None, summary_writer=None, 
                scheduler=None, stochastic=True, early_stopping_patience=10):
    """
    Trains the fraud detection model.

    Args:
        model (torch.nn.Module): The model to train.
        num_epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model ('cuda' or 'cpu').
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        early_stopping_patience (int): Number of epochs to wait before stopping if validation loss does not improve.

    Returns:
        tuple: Lists of training loss, training accuracy, validation accuracy, and final test metrics.
    """
    
    start_time = time.time()
    minibatch_loss_list, train_metrics_list, valid_metrics_list = [], [], []
    best_valid_f1 = 0.0  # Track Best F1-score
    patience_counter = 0  # Track early stopping criteria
    
    if summary_writer is not None:
        summary_writer = SummaryWriter()

    print("Starting Training...")
    # log_interval = 500  # Adjust log interval for fraud dataset
    log_interval = max(10, len(train_loader) // 10)
    model.train()

    for epoch in range(num_epochs):
        if stochastic:
            model.resample_dropout_masks(next(iter(train_loader))[0])
        model.train()
        
        total_loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            targets = targets.view(-1, 1)  

            optimizer.zero_grad()
            logits = model(features, apply_mask=True).to(device)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            minibatch_loss_list.append(loss.item())

            if batch_idx % log_interval == 0:
                print(f"[Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            valid_metrics = compute_metrics(model, valid_loader, device)
            train_metrics = compute_metrics(model, train_loader, device)
            
            # Add loss to the metrics dictionaries
            train_metrics['loss'] = avg_train_loss
            
            # Calculate validation loss
            valid_loss = 0
            for features, targets in valid_loader:
                features, targets = features.to(device), targets.to(device)
                targets = targets.view(-1, 1)
                logits = model(features).to(device)
                loss = criterion(logits, targets)
                valid_loss += loss.item()
            
            avg_valid_loss = valid_loss / len(valid_loader)
            valid_metrics['loss'] = avg_valid_loss

            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                  f"Valid Loss: {avg_valid_loss:.4f} | "
                  f"Valid Acc: {valid_metrics['accuracy']:.2f}% | "
                  f"Valid Precision: {valid_metrics['precision']:.2f}% | "
                  f"Valid Recall: {valid_metrics['recall']:.2f}% | "
                  f"Valid F1-score: {valid_metrics['f1_score']:.2f}% | "
                  f"Valid AUC-PR: {valid_metrics['auc_pr']:.2f}%")

            train_metrics_list.append(train_metrics)
            valid_metrics_list.append(valid_metrics)

        elapsed = (time.time() - start_time)/60
        print(f"Time Elapsed: {elapsed:.2f} minutes")

        # Early Stopping Check (Using F1-score)
        if valid_metrics['f1_score'] > best_valid_f1:
            best_valid_f1 = valid_metrics['f1_score']
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved. Saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered. Training stopped.")
                break

        if scheduler:
            scheduler.step(valid_metrics['f1_score'])
            
    # Load the best model before testing
    model.load_state_dict(torch.load("best_model.pth"))
    print("Evaluating best saved model on test data.")

    # Final Test Evaluation
    test_metrics = compute_metrics(model, test_loader, device)
    print(f"**Final Test Results:** "
          f"Accuracy: {test_metrics['accuracy']:.2f}%, "
          f"Precision: {test_metrics['precision']:.2f}%, "
          f"Recall: {test_metrics['recall']:.2f}%, "
          f"F1-score: {test_metrics['f1_score']:.2f}%, "
          f"AUC-PR: {test_metrics['auc_pr']:.2f}%")

    return minibatch_loss_list, train_metrics_list, valid_metrics_list, test_metrics
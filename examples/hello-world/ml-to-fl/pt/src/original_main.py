import os
import torch
import torch.nn as nn
import torch.optim as optim
from net import FraudNet, AttentionTransformerFraudNet, EnhancedFraudNet  # Import fraud detection model
from data import get_dataloaders_fraud, get_dataloaders_fraud_2  # Import dataset functions
from evaluation import evaluate_model  # Import evaluation function
from train import train_model, set_all_seeds  # Import training function from train.py
import pandas as pd
import sys
from plot import plot_metrics, plot_confusion_matrices, plot_aucpr
import pickle

def main():
    # Load fraud dataset
    set_all_seeds(42)
    
    # Set dataset path
    DATASET_PATH = "/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/creditcard.csv"
    TEST_DATASET_PATH = "/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/creditcard.csv"
    TRAIN_VALID_DATASET_PATH = "/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/train_valid.csv"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Training hyperparameters
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.00003
    
    
    df = pd.read_csv(DATASET_PATH)
    input_size = df.shape[1] - 1
    print(f"Detected input size: {input_size}")
    
    model = EnhancedFraudNet(input_size=input_size).to(DEVICE)
    model_name = model.__class__.__name__
    
    save_plot_dir = f'plot_{model_name}_{batch_size}_{num_epochs}_{learning_rate}'
    
    # train_loader, valid_loader, test_loader, class_weights = get_dataloaders_fraud(
    #     DATASET_PATH, batch_size=batch_size, use_smote=True, plot=True, save_plot_dir=save_plot_dir
    # )
    
    train_loader, valid_loader, test_loader, class_weights, _ = get_dataloaders_fraud_2(
        TRAIN_VALID_DATASET_PATH, test_csv=TEST_DATASET_PATH, batch_size=batch_size, use_smote=True, plot=True, save_plot_dir=save_plot_dir
    )
    
    class_weights = class_weights
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=DEVICE)
    
    # Loss Function (No weight balancing since using SMOTE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Call `train.py` instead of writing the training loop here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, verbose=True
    )
    
    train_loss_list, train_metrics_list, valid_metrics_list, test_metrics = train_model(
        model, num_epochs, train_loader, valid_loader, test_loader, optimizer,
        criterion, DEVICE, scheduler=scheduler, stochastic=False
    )
    
    # Create metrics directory if it doesn't exist
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save training metrics
    metrics_data = {
        'train_metrics': train_metrics_list,
        'valid_metrics': valid_metrics_list,
        'test_metrics': test_metrics,
        'train_loss': train_loss_list
    }
    
    metrics_file = os.path.join(metrics_dir, f"{model_name}_{batch_size}_{num_epochs}_{learning_rate}_metrics.pickle")
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics_data, f)
    
    print(f"Metrics saved to {metrics_file}")
    
    # Create plots
    plot_metrics(train_metrics_list, fig_name="Training Metrics", save_path=f"{save_plot_dir}/train_metrics.png")
    plot_metrics(valid_metrics_list, fig_name="Validation Metrics", save_path=f"{save_plot_dir}/valid_metrics.png")
    plot_confusion_matrices(model, test_loader, threshold=0.85, save_path=f"{save_plot_dir}/confusion_matrix.png")
    plot_aucpr(model, test_loader, device=DEVICE, save_path=f"{save_plot_dir}/auc_pr.png")
    
    # Save the trained model
    best_model_path = "best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model from training phase.")
    
    # Save the best model explicitly at a clear location for future usage
    final_model_path = f"./best_{model_name}_{batch_size}_{num_epochs}_{learning_rate}_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final best model saved explicitly at {final_model_path}")
    
    # Evaluate model
    print("Evaluating Model on Test Set...")
    evaluate_model(model, test_loader, DEVICE)
    
    return metrics_data  # Return metrics data for potential further use

if __name__ == "__main__":
    main()

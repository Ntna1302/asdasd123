import os
import torch
import torch.nn as nn
import torch.optim as optim
from net import FraudNet  # Import fraud detection model
from data import get_dataloaders_fraud  # Import dataset functions
from evaluation import evaluate_model  # Import evaluation function
from train import train_model  # Import training function from train.py
import pandas as pd
import sys
from plot import plot_metrics, plot_confusion_matrices
def main():
    # Load fraud dataset

    # Set dataset path
    DATASET_PATH = "/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/creditcard.csv"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Training hyperparameters
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.00003

    save_plot_dir = 'plot'

    train_loader, valid_loader, test_loader = get_dataloaders_fraud(
        DATASET_PATH, batch_size=batch_size, use_smote=True, plot=True, save_plot_dir=save_plot_dir
    )

    df = pd.read_csv(DATASET_PATH)
    input_size = df.shape[1] - 1  
    print(f"Detected input size: {input_size}")

    # Initialize Model
    model = FraudNet(input_size=input_size).to(DEVICE)

    # Loss Function (No weight balancing since using SMOTE)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Call `train.py` instead of writing the training loop here

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, verbose=True
    )

    train_loss_list, train_metrics_list, valid_metrics_list, test_metrics = train_model(
        model, num_epochs, train_loader, valid_loader, test_loader, optimizer,
        criterion, DEVICE, scheduler=scheduler , stochastic=False
    )
    
    plot_metrics(train_metrics_list, fig_name="train_metrics", save_path=f"{save_plot_dir}/train_metrics.png")
    plot_metrics(valid_metrics_list, fig_name="valid_metrics", save_path=f"{save_plot_dir}/valid_metrics.png")
    plot_confusion_matrices(model, test_loader, threshold=0.85, save_path=f"{save_plot_dir}/confusion_matrix.png")
    
    # Save the trained model
    best_model_path = "best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model from training phase.")

    # Save the best model explicitly at a clear location for future usage
    final_model_path = "./best_fraud_model.pth"
    torch.save(model.state_dict(), best_model_path)
    print(f"Final best model saved explicitly at {final_model_path}")
    
    # Evaluate model
    print("Evaluating Model on Test Set...")
    evaluate_model(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()

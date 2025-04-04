import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek 
from plot import *
import os

class FraudDataset(Dataset):
    """Custom PyTorch Dataset for Fraud Detection"""
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def check_data_leakage(X_train, X_valid, X_test):
    train_df = pd.DataFrame(X_train)
    valid_df = pd.DataFrame(X_valid)
    test_df = pd.DataFrame(X_test)

    overlap_train_test = test_df.merge(train_df, how="inner")
    overlap_test_valid = test_df.merge(valid_df, how="inner")
    overlap_train_valid = train_df.merge(valid_df, how="inner")
    
    print(f"Overlapping samples between Train & Test: {len(overlap_train_test)}")
    print(f"Overlapping samples between Validation & Test: {len(overlap_test_valid)}")
    print(f"Overlapping samples between Validation & Train: {len(overlap_train_valid)}")


def add_noise(X_train, mean=0, sigma=0.1):
    """
    Adds Gaussian noise to the training data.
    Args:
        X_train (numpy array): Training features.
        mean (float): Mean of Gaussian noise.
        sigma (float): Standard deviation of noise.

    Returns:
        numpy array: Noisy training features.
    """
    noise = np.random.normal(mean, sigma, X_train.shape)
    return X_train + noise

def get_dataloaders_fraud(csv_path, batch_size=64, num_workers=0, use_smote=True, add_noise_flag=True, noise_std=0.1, plot = False, save_plot_dir = '.'):
    # Load dataset from CSV
    df = pd.read_csv(csv_path)
    
    if plot:
        os.makedirs(save_plot_dir, exist_ok=True)
        plot_distribution_org(df, save_path=f'{save_plot_dir}/class_distribution.png')
    
    df.drop_duplicates(inplace=True)
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Extract features & labels
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    ro_scaler = RobustScaler()
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y.tolist())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # First, split into train (64%), validation (16%), and test (20%)
    # This ensures all sets have the same original distribution
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.36, stratify=y)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.2/0.36, stratify=y_temp)
    
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)
    X_test = ro_scaler.transform(X_test)

    # Apply SMOTE only to training set
    if use_smote:
        print("Applying SMOTE to balance training data...")
        y_train_before = y_train
        smote = SMOTETomek(sampling_strategy=0.3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        if plot:
            plot_before_after_smote(y_train_before, y_train, save_path=f'{save_plot_dir}/class_distribution_smote.png')
    
    if plot:
        plot_distribution_train_valid_test(y_train, y_valid, y_test, save_path=f'{save_plot_dir}/class_distribution_split.png')
    
    # Add Gaussian Noise to Training Data (After SMOTE)
    if add_noise_flag:
        print(f"Adding Gaussian noise (std={noise_std}) to training data...")
        X_train = add_noise(X_train, sigma=noise_std)
    
    # Ensure variables exist before calling check_data_leakage()
    if X_valid is not None and X_test is not None:
        check_data_leakage(X_train, X_valid, X_test)
    
    # Convert labels to PyTorch-compatible format
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Create PyTorch Datasets
    train_dataset = FraudDataset(X_train, y_train)
    valid_dataset = FraudDataset(X_valid, y_valid)
    test_dataset = FraudDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    print(f"Training set size after SMOTE: {len(train_dataset)} samples")
    print(f"Validation set size: {len(valid_dataset)} samples")
    print(f"Test set size: {len(test_dataset)} samples")
    
    return train_loader, valid_loader, test_loader
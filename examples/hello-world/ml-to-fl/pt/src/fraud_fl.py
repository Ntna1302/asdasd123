# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

from net import FraudNet, EnhancedFraudNet, AttentionTransformerFraudNet
from train import *
from evaluation import *
from data import get_dataloaders_fraud
from plot import plot_metrics, plot_confusion_matrices, plot_aucpr

# (1) import nvflare client API
import nvflare.client as flare
# (optional) metrics
from nvflare.client.tracking import SummaryWriter

# (optional) set a fix place so we don't need to download everytime
# Use site-specific dataset based on site name
BASE_PATH = "/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data"

# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0"
batch_size = 64
learning_rate = 0.0003

def get_dataset_path(site_name):
    """
    Determine the dataset path based on the site name.
    """
    if site_name == "site-1":
        return os.path.join(BASE_PATH, "site1.csv")
    elif site_name == "site-2":
        return os.path.join(BASE_PATH, "site2.csv")
    elif site_name == "server":
        return os.path.join(BASE_PATH, "server.csv")
    else:
        # Default fallback to the original dataset
        print(f"Warning: Unknown site name '{site_name}', using original dataset")
        return os.path.join(BASE_PATH, "creditcard.csv")

def main():
    set_all_seeds(42)  # Comment out this so that it is pure randomness when training.
    epochs = 10
    save_plot_dir = 'plot'
    
    # (2) initializes NVFlare client API
    flare.init()
    
    # Get the site name from NVFlare
    site_name = flare.get_site_name()
    print(f"Running as site: {site_name}")
    
    # Get the dataset path for this site
    dataset_path = get_dataset_path(site_name)
    print(f"Using dataset: {dataset_path}")
    
    # Load site-specific data
    train_loader, valid_loader, test_loader, class_weights = get_dataloaders_fraud(
        dataset_path, batch_size=batch_size, use_smote=False, plot=True, 
        save_plot_dir=f"{save_plot_dir}/{site_name}"
    )
    
    class_weights = class_weights
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=DEVICE)
    net = FraudNet(device=DEVICE)
    
    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {site_name}]\n")
        
        # (4) loads model from NVFlare
        net.load_state_dict(input_model.params)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, verbose=True
        )
        
        # (optional) use GPU to speed things up
        net.to(DEVICE)
        
        train_loss_list, train_metrics_list, valid_metrics_list, test_metrics = train_model(
            model=net, num_epochs=epochs, train_loader=train_loader, valid_loader=valid_loader, 
            test_loader=test_loader, optimizer=optimizer, criterion=criterion, device=DEVICE, 
            scheduler=scheduler, stochastic=False
        )
        
        # Create site-specific directories for plots
        site_plot_dir = f"{save_plot_dir}/{site_name}"
        os.makedirs(site_plot_dir, exist_ok=True)
        
        plot_metrics(train_metrics_list, fig_name=f"{site_name}_train_metrics", save_path=f"{site_plot_dir}/train_metrics.png")
        plot_metrics(valid_metrics_list, fig_name=f"{site_name}_valid_metrics", save_path=f"{site_plot_dir}/valid_metrics.png")
        plot_confusion_matrices(net, test_loader, threshold=0.85, save_path=f"{site_plot_dir}/confusion_matrix.png")
        plot_aucpr(net, test_loader, device=DEVICE, save_path=f"{site_plot_dir}/auc_pr.png")
        
        print("Finished Training")
        PATH = f"./{site_name}_fraud_net_fl.pth"
        torch.save(net.state_dict(), PATH)
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics=test_metrics,
        )
        # (8) send model back to NVFlare
        flare.send(output_model)
        print("Finished Sending Model")

if __name__ == "__main__":
    main()
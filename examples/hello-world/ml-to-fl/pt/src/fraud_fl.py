# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from net import FraudNet
from train import *
from evaluation import *
from data import get_dataloaders_fraud


# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

# (optional) set a fix place so we don't need to download everytime
DATASET_PATH = "/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/creditcard.csv"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0"
batch_size = 64
learning_rate = 0.0003

def main():
    set_all_seeds(42)
    
    epochs = 10
    
    train_loader, valid_loader, test_loader = get_dataloaders_fraud(
        DATASET_PATH, batch_size=batch_size, use_smote=True
    )

    net = FraudNet(device = DEVICE)

    # (2) initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")

        # (4) loads model from NVFlare
        net.load_state_dict(input_model.params)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='max', patience=3, verbose=True
                        )
        # (optional) use GPU to speed things up
        net.to(DEVICE)
        
        train_loss_list, train_metrics_list, valid_metrics_list, test_metrics = train_model(
        model = net, num_epochs = epochs, train_loader=train_loader , valid_loader=valid_loader, test_loader=test_loader, optimizer = optimizer,
        criterion = criterion, device = DEVICE, scheduler=scheduler , stochastic=False
        )
        print("Finished Training")
        
        PATH = "./fraud_net_fl.pth"
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

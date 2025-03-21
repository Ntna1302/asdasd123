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
import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
class FraudNet(nn.Module):
    def __init__(self, input_size=30, dropout_rate=0.6, device=None):
        """
        A fully connected neural network (MLP) for fraud detection.

        Args:
            input_size (int): Number of features in the dataset.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(FraudNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)  # Binary classification output (logits)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

        # Placeholder for stochastic dropout masks
        self.dense_mask = None

    def forward(self, x, apply_mask=False):
        """
        Forward pass through the fraud detection model.

        Args:
            x (torch.Tensor): Input features.
            apply_mask (bool, optional): Whether to apply a dropout mask for Federated Learning. Default is False.

        Returns:
            torch.Tensor: Logits for binary classification (sigmoid will be applied in loss function).
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        if apply_mask and self.dense_mask is not None:
            self.dense_mask = self.dense_mask.to(x.device)
            x = x * self.dense_mask
            x = x / (1 - self.dropout_rate)  # Normalize output

        x = self.fc4(x)  # No sigmoid here! BCEWithLogitsLoss expects raw logits
        return x

    def resample_dropout_masks(self, x):
        """
        Resample dropout mask for fully connected layers (used in Stochastic Dropout).

        Args:
            x (torch.Tensor): Input tensor.
        """
        self.dense_mask = torch.bernoulli(torch.ones(self.fc3.out_features) * (1 - self.dropout_rate)).to(x.device)


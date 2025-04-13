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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FraudNet(nn.Module):
    def __init__(self, input_size=30, dropout_rate=0.1, device=None):
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


class EnhancedFraudNet(nn.Module):
    def __init__(self, input_size=30, dropout_rate=0.2, device=None):
        """
        A fully connected neural network (MLP) for fraud detection with enhancements.

        Args:
            input_size (int): Number of features in the dataset.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(EnhancedFraudNet, self).__init__()

        # Define the network layers
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 1)  # Output layer for binary classification

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

        # Stochastic dropout mask (used for federated learning)
        self.dense_mask = None

        # Optionally, use LeakyReLU for non-linearity
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x, apply_mask=False):
        """
        Forward pass through the fraud detection model with enhanced regularization.

        Args:
            x (torch.Tensor): Input features.
            apply_mask (bool, optional): Whether to apply a dropout mask for Federated Learning. Default is False.

        Returns:
            torch.Tensor: Logits for binary classification (no sigmoid applied here, since BCEWithLogitsLoss expects raw logits).
        """
        # First fully connected layer with batch normalization
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout
        
        # Second fully connected layer with batch normalization
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        
        # Third fully connected layer with batch normalization
        x = self.leaky_relu(self.bn3(self.fc3(x)))

        # Apply stochastic dropout mask (optional, only in federated learning)
        if apply_mask and self.dense_mask is not None:
            self.dense_mask = self.dense_mask.to(x.device)
            x = x * self.dense_mask
            x = x / (1 - self.dropout_rate)  # Normalize output after dropout

        # Output layer (no sigmoid activation, because we use BCEWithLogitsLoss)
        x = self.fc4(x)
        return x

    def resample_dropout_masks(self, x):
        """
        Resample the dropout mask for fully connected layers (used in Stochastic Dropout).

        Args:
            x (torch.Tensor): Input tensor.
        """
        # Resample the dropout mask with a Bernoulli distribution
        self.dense_mask = torch.bernoulli(torch.ones(self.fc3.out_features) * (1 - self.dropout_rate)).to(x.device)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing feature interactions."""
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # Scaled dot-product attention
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class TabTransformerBlock(nn.Module):
    """Transformer block adapted for tabular data."""
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TabTransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        
        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class FeatureEmbedder(nn.Module):
    """Transforms raw features into embeddings for transformer."""
    def __init__(self, input_size, embed_size):
        super(FeatureEmbedder, self).__init__()
        self.embedder = nn.Linear(input_size, embed_size)
        
    def forward(self, x):
        return self.embedder(x).unsqueeze(1)  # Add sequence dimension

class AttentionTransformerFraudNet(nn.Module):
    """
    Advanced neural network for fraud detection combining transformer architecture 
    with traditional deep learning techniques.
    
    Args:
        input_size (int): Number of features in the dataset
        embed_size (int): Dimension of feature embeddings
        heads (int): Number of attention heads
        transformer_layers (int): Number of transformer blocks
        dropout_rate (float): Dropout probability for regularization
        forward_expansion (int): Expansion factor for feed-forward network
        device (torch.device): Device to run the model on
    """
    def __init__(self, 
                 input_size=30, 
                 embed_size=64, 
                 heads=4, 
                 transformer_layers=2,
                 dropout_rate=0.1, 
                 forward_expansion=2,
                 device=None):
        super(AttentionTransformerFraudNet, self).__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_embedder = FeatureEmbedder(input_size, embed_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TabTransformerBlock(embed_size, heads, dropout_rate, forward_expansion) 
             for _ in range(transformer_layers)]
        )
        
        # Traditional neural network path
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Combine both paths
        self.combiner = nn.Linear(embed_size + 64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)  # Binary classification
        
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.dense_mask = None
        
    def forward(self, x, apply_mask=False):
        """
        Forward pass through the enhanced fraud detection model.
        
        Args:
            x (torch.Tensor): Input features
            apply_mask (bool): Whether to apply dropout mask for Federated Learning
            
        Returns:
            torch.Tensor: Logits for binary classification
        """
        batch_size = x.shape[0]
        
        # Transformer path
        embedded = self.feature_embedder(x)
        
        for transformer in self.transformer_blocks:
            embedded = transformer(embedded, embedded, embedded)
            
        # Extract the transformer output (squeeze the sequence dimension)
        transformer_out = embedded.squeeze(1)
        
        # Traditional path
        traditional = F.relu(self.bn1(self.fc1(x)))
        traditional = self.dropout(traditional)
        traditional = F.relu(self.bn2(self.fc2(traditional)))
        
        # Combine paths
        combined = torch.cat([transformer_out, traditional], dim=1)
        out = F.relu(self.bn3(self.combiner(combined)))
        
        if apply_mask and self.dense_mask is not None:
            self.dense_mask = self.dense_mask.to(out.device)
            out = out * self.dense_mask
            out = out / (1 - self.dropout_rate)  # Normalize output
            
        out = self.fc4(out)  # Raw logits
        return out
    
    def resample_dropout_masks(self, x):
        """
        Resample dropout mask for fully connected layers (used in Stochastic Dropout).
        
        Args:
            x (torch.Tensor): Input tensor
        """
        self.dense_mask = torch.bernoulli(torch.ones(32) * (1 - self.dropout_rate)).to(x.device)
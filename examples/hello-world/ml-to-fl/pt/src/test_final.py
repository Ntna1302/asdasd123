from net import FraudNet, EnhancedFraudNet, AttentionTransformerFraudNet
from data import get_dataloaders_fraud
from evaluation import *
from train import set_all_seeds
import torch
from torch.utils.data import ConcatDataset, DataLoader

# Create model instance first
model = FraudNet()
# Load state dictionary from saved file
model_path = 'examples/hello-world/ml-to-fl/pt/na132/jobs/workdir/server/simulate_job/app_server/best_FL_global_model.pt'
model_path = 'examples/hello-world/ml-to-fl/pt/na132/jobs/workdir/site-1/site-1_fraud_net_fl.pth'
state_dict = torch.load(model_path)
if 'model' in state_dict:
    model.load_state_dict(state_dict['model'])
else:
    model.load_state_dict(state_dict)
# Move model to the correct device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
# Training hyperparameters
batch_size = 32
DATASET_PATH = "examples/hello-world/ml-to-fl/pt/src/data/server.csv"
set_all_seeds(42) 
# Get the three separate loaders
train_loader, valid_loader, test_loader, _ = get_dataloaders_fraud(
    DATASET_PATH, batch_size=batch_size, use_smote=False
)

# Combine all three datasets
combined_dataset = ConcatDataset([
    train_loader.dataset,
    valid_loader.dataset,
    test_loader.dataset
])

# Create a new combined loader
combined_loader = DataLoader(
    combined_dataset, 
    batch_size=batch_size,
    shuffle=True
)

print(f"Combined dataset size: {len(combined_dataset)}")

# Evaluate the model on the combined dataset
evaluate_model(model, combined_loader, device=DEVICE, threshold=0.85)
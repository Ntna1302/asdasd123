from net import FraudNet
from data import get_dataloaders_fraud
from evaluation import *
import torch

# Create model instance first
model = FraudNet()

# Load state dictionary from saved file
model_path = 'examples/hello-world/ml-to-fl/pt/na132/jobs/workdir/site-1/fraud_net_fl.pth'
state_dict = torch.load(model_path)

# Load the state dictionary into your model
model.load_state_dict(state_dict)

# Move model to the correct device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)

# Training hyperparameters
batch_size = 32
DATASET_PATH = "examples/hello-world/ml-to-fl/pt/src/data/creditcard.csv"
train_loader, valid_loader, test_loader = get_dataloaders_fraud(
    DATASET_PATH, batch_size=batch_size, use_smote=False
)

# Evaluate the model
evaluate_model(model, test_loader, device=DEVICE, threshold=0.5)
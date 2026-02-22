# Placeholder file for model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrustScoreModel(nn.Module):
    def __init__(self, input_dim):
        super(TrustScoreModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output trust score between 0 and 1
        return x

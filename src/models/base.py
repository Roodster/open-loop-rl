import torch.nn as nn



class BaseModel(nn.Module):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.in_representation = in_representation
        self.hidden_features = hidden_features
        
        self.reward = nn.Linear(self.hidden_features, 1).to(self.device)
        self.value = nn.Linear(self.hidden_features, 1).to(self.device)
        
    def forward(self):
        pass
    
    
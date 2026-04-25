import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreatMLP_Basic(nn.Module):
    def __init__(self, num_uav_types, embed_dim=4):
        super().__init__()
        
        # UAV type embedding
        self.embedding = nn.Embedding(num_uav_types, embed_dim)
        
        # Input size: x, y, speed + embedding
        input_dim = 3 + embed_dim
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        
        # Outputs
        self.threat_head = nn.Linear(32, 5)   # 5 threat levels
        self.damage_head = nn.Linear(32, 1)   # expected damage

    def forward(self, x, y, speed, uav_type):
        # Embed UAV type
        uav_embed = self.embedding(uav_type)
        
        # Concatenate inputs
        inputs = torch.cat([x, y, speed, uav_embed], dim=1)
        
        # MLP
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        # Outputs
        threat_logits = self.threat_head(h)
        threat_probs = F.softmax(threat_logits, dim=1)
        
        damage = torch.sigmoid(self.damage_head(h))
        
        return threat_probs, damage
    
class ThreatMLP_Trajectory(nn.Module):
    def __init__(self, num_uav_types, embed_dim=4):
        super().__init__()
        
        self.embedding = nn.Embedding(num_uav_types, embed_dim)
        
        # Input: x, y, speed, dx, dy + embedding
        input_dim = 5 + embed_dim
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.threat_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

        self.damage_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, y, speed, dx, dy, uav_type):
        uav_embed = self.embedding(uav_type)
        
        inputs = torch.cat([x, y, speed, dx, dy, uav_embed], dim=1)
        
        h = self.shared(inputs)
        
        threat_logits = self.threat_head(h)
        threat_probs = F.softmax(threat_logits, dim=1)
        
        damage = torch.sigmoid(self.damage_head(h))
        
        return threat_probs, damage

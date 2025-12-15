"""
@author: Duyen Nguyen <DuyenNHCE200017@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    Mạng Actor-Critic
    Input: Trạng thái game
    Output:
        - Actor: Xác suất chọn hành động (Policy) - Ở đây là nhảy hoặc không nhảy
        - Critic: Giá trị của trạng thái hiện tại (Value)
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head: Đưa ra xác suất hành động
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # Softmax để chuyển đổi thành xác suất
        )
        
        # Critic head: Đưa ra giá trị ước lượng của trạng thái
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass của mạng Actor-Critic
        x: Tensor chứa trạng thái
        """
        # Đưa x qua lớp feature chung
        features = self.feature_layer(x)
        
        # Tính toán Policy (xác suất hành động)
        policy = self.actor(features)
        
        # Tính toán Value (giá trị trạng thái)
        value = self.critic(features)
        
        return policy, value
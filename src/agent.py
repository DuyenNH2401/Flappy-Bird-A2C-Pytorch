"""
@author: Duyen Nguyen <DuyenNHCE200017@gmail.com>
"""
import torch
import torch.optim as optim
import numpy as np
from network import ActorCritic

class A2CAgent:
    """
    Agent A2C (Advantage Actor-Critic)
    """
    def __init__(self, input_dim, hidden_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma # Hệ số chiết khấu cho phần thưởng tương lai
        
        # input_dim: kích thước trạng thái
        # hidden_dim: số lượng neuron ẩn
        # action_dim: số lượng hành động
        self.model = ActorCritic(input_dim, hidden_dim, action_dim)
        
        # Dùng Adam optimizer để tối ưu hóa mạng
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        """
        # Chuyển state từ numpy array sang torch tensor
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Đưa vào mô hình để lấy xác suất và giá trị
        probs, state_value = self.model(state)
        
        # Tạo distribution để sample hành động
        dist = torch.distributions.Categorical(probs)
        action = dist.sample() # Chọn ngẫu nhiên hành động dựa trên xác suất
        
        # Lấy log probability của hành động vừa chọn
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, state_value

    def update(self, rewards, log_probs, state_values, next_state_value, dones):
        """
        Cập nhật mạng dựa trên dữ liệu thu thập được.
        """
        # Tính toán Discounted Returns (R)
        # R_t = r_t + gamma * R_{t+1}
        returns = []
        R = next_state_value # Giá trị dự đoán của trạng thái cuối cùng
        
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done: 
                R = 0 # Nếu game over thì giá trị tương lai là 0
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # Chuyển đổi sang tensor
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        state_values = torch.stack(state_values).squeeze()
        
        # Tính toán Advantage
        # Advantage = Returns thực tế - Giá trị dự đoán (Critic)
        # A(s, a) = R - V(s)
        advantages = returns - state_values
        # Chuẩn hóa Advantage để ổn định quá trình huấn luyện
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        # --- Tính Loss ---
        
        # 1. Actor Loss (Policy Loss)
        # Loss = -sum(log_prob * advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # 2. Critic Loss (Value Loss)
        # Loss = MSE(Returns, Values)
        critic_loss = (advantages.pow(2)).mean()
        
        # Tổng Loss
        loss = actor_loss + 0.5 * critic_loss 
        
        # Tối ưu hóa
        self.optimizer.zero_grad() # Xóa gradient cũ
        loss.backward()            # Tính gradient mới
        self.optimizer.step()      # Cập nhật trọng số
        
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval() 

"""
@author: Duyen Nguyen <DuyenNHCE200017@gmail.com>
"""
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import time
import sys
import os

# Add current directory to path so we can import agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import A2CAgent

def test():
    # Cấu hình giống lúc train
    HIDDEN_DIM = 256
    ACTION_DIM = 2
    
    # Lưu ý: Phải để use_lidar giống lúc train
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    
    # Tự động lấy kích thước
    INPUT_DIM = env.observation_space.shape[0]
    print(f"Observation Space: {INPUT_DIM}")
    
    # Khởi tạo Agent
    agent = A2CAgent(INPUT_DIM, HIDDEN_DIM, ACTION_DIM)
    
    # Load model
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "a2c_flappy_bird.pth")
        agent.load(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found! Please run train.py first.")
        return

    # Run 5 episodes
    for i in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"Starting Game {i+1}...")
        
        while not done:
            # Select action
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            
            # time.sleep(0.01) 
            
        print(f"Game {i+1} finished. Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    test()

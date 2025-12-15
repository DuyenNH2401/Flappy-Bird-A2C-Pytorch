# @author: Duyen Nguyen <DuyenNHCE200017@gmail.com>

import gymnasium as gym
import flappy_bird_gymnasium
import torch
import numpy as np
import os
from agent import A2CAgent

def train():
    # Cấu hình
    ENV_NAME = "FlappyBird-v0"
    MAX_EPISODES = 50000
    LOG_INTERVAL = 1000           # In kết quả sau mỗi 1000 màn
    HIDDEN_DIM = 256             # Số lượng neuron ẩn
    ACTION_DIM = 2               # 0: Không làm gì, 1: Nhảy
    LEARNING_RATE = 0.0003

    # Khởi tạo môi trường
    # use_lidar=False -> 12 features
    # use_lidar=True  -> 180 features
    env = gym.make(ENV_NAME, use_lidar=False) 
    
    # Tự động lấy kích thước đầu vào từ môi trường
    INPUT_DIM = env.observation_space.shape[0]
    
    # Khởi tạo Agent
    agent = A2CAgent(INPUT_DIM, HIDDEN_DIM, ACTION_DIM, lr=LEARNING_RATE)
    
    print("Starting training...")
    running_reward = 0
    
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        
        rewards = []
        log_probs = []
        state_values = []
        
        done = False
        while not done:
            action, log_prob, state_value = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            log_probs.append(log_prob)
            state_values.append(state_value)
            
            state = next_state
            
            if done:
                _, _, next_state_value = agent.select_action(state)
                
                loss = agent.update(rewards, log_probs, state_values, next_state_value, [done]*len(rewards))
                
                total_reward = sum(rewards)
                
                running_reward = 0.05 * total_reward + (1 - 0.05) * running_reward
                
                if episode % LOG_INTERVAL == 0:
                    print(f"Episode {episode}\tLast Reward: {total_reward:.2f}\tAvg Reward: {running_reward:.2f}\tLoss: {loss:.4f}")
                    # Save model
                    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    agent.save(os.path.join(model_dir, "a2c_flappy_bird.pth"))

if __name__ == "__main__":
    train()

'''
model : 4 -> 1 action
DQN :
        predicts = torch.gather(self.model(history), 1, actions)
        max_next_q = self.target_model(next_history).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q

        loss = nn.SmoothL1Loss()(predicts, targets).to(self.device)
'''       

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs = (4,     
       


class MyModel
    nn 

class MyAgent:



2025-02-07

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

class HybridWalkingPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head for RL
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head for RL
        self.value = nn.Linear(hidden_dim, 1)
        
        # Behavior cloning head
        self.bc_policy = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        features = self.backbone(state)
        return features
    
    def get_action(self, state, imitation_weight=0.5):
        features = self.forward(state)
        
        # RL action distribution
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_std).expand_as(action_mean)
        rl_dist = Normal(action_mean, action_std)
        
        # Behavior cloning action
        bc_action = torch.tanh(self.bc_policy(features))
        
        # Combine both actions
        rl_action = torch.tanh(rl_dist.sample())
        combined_action = imitation_weight * bc_action + (1 - imitation_weight) * rl_action
        
        return combined_action
    
    def get_value(self, state):
        features = self.forward(state)
        return self.value(features)

class HybridWalkingTrainer:
    def __init__(self, policy, lr=3e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
    def compute_hybrid_loss(self, states, actions, rewards, next_states, dones, 
                           expert_actions, bc_weight=0.5, gamma=0.99):
        # Behavior Cloning Loss
        features = self.policy.forward(states)
        bc_actions = torch.tanh(self.policy.bc_policy(features))
        bc_loss = nn.MSELoss()(bc_actions, expert_actions)
        
        # RL Loss (PPO style)
        values = self.policy.get_value(states)
        next_values = self.policy.get_value(next_states)
        
        # Compute advantages
        advantages = []
        returns = []
        running_return = 0
        
        for r, d, nv in zip(reversed(rewards), reversed(dones), reversed(next_values)):
            running_return = r + gamma * running_return * (1-d)
            returns.insert(0, running_return)
            
        returns = torch.tensor(returns)
        advantages = returns - values
        
        # Policy loss
        action_mean = self.policy.policy_mean(features)
        action_std = torch.exp(self.policy.policy_std)
        dist = Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Combined loss
        total_loss = (1 - bc_weight) * (policy_loss + 0.5 * value_loss) + bc_weight * bc_loss
        
        return total_loss
    
    def train_step(self, batch):
        states, actions, rewards, next_states, dones, expert_actions = batch
        loss = self.compute_hybrid_loss(states, actions, rewards, next_states, dones, expert_actions)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def collect_expert_demonstrations(env, expert_policy, num_episodes=100):
    expert_data = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = expert_policy(state)  # Your expert policy here
            next_state, reward, done, _ = env.step(action)
            expert_data.append((state, action))
            state = next_state
            
    return expert_data




# 환경과 정책 초기화
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy = HybridWalkingPolicy(state_dim, action_dim)
trainer = HybridWalkingTrainer(policy)

# 전문가 데모 수집
expert_data = collect_expert_demonstrations(env, expert_policy)

# 학습 루프
for epoch in range(num_epochs):
    # 배치 데이터 준비
    batch = prepare_batch(env, policy, expert_data)
    loss = trainer.train_step(batch)
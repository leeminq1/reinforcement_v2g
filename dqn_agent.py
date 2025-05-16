# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size=83, output_size=20):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, output_size),
            nn.Tanh()  # 출력을 -1~1 범위로 제한
        )
        
        # Xavier 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=83, action_size=20):  # 상태 크기도 83으로 수정
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN 하이퍼파라미터
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # 할인율
        self.epsilon = 1.0   # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Q-네트워크와 타겟 네트워크
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return np.random.uniform(-1, 1, (self.action_size,))
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return action_values.cpu().data.numpy()[0]
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 배치 샘플링
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 배치 데이터 준비
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.FloatTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        
        # 현재 Q 값 계산
        current_q_values = self.model(states)
        
        # 다음 상태의 최대 Q 값 계산
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 손실 계산 및 최적화
        loss = 0
        for i in range(self.action_size):
            action_mask = (actions[:, i] != 0).float()
            q_value = current_q_values[:, i]
            loss += torch.mean(action_mask * (q_value - target_q_values) ** 2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 입실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict()) 
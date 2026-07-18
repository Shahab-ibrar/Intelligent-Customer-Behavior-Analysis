import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Any, Optional

class CustomerEnv:
    """
    Simulation Environment for Reinforcement Learning.
    State representation: [Recency, Frequency, Monetary, PC1, PC2]
    Actions:
      - 0: No Action
      - 1: 10% Coupon
      - 2: Premium Trial
    """
    def __init__(self, df: pd.DataFrame, regressor: Any, max_steps: int = 10):
        self.df = df.copy()
        self.regressor = regressor
        self.max_steps = max_steps
        self.current_step = 0
        self.state = np.zeros(5, dtype=np.float32)
        
        # Adjustments and Costs
        self.prob_adj = {0: 0.30, 1: 0.55, 2: 0.75}
        self.costs = {0: 0, 1: 1, 2: 5}
        
    def reset(self, customer_id: Optional[int] = None) -> np.ndarray:
        """
        Resets the environment. If customer_id is provided, starts with that customer's state,
        otherwise samples a random customer from the dataset.
        """
        if customer_id is None:
            row = self.df.sample(n=1).iloc[0]
        else:
            subset = self.df[self.df['CustomerID'] == customer_id]
            if len(subset) == 0:
                row = self.df.sample(n=1).iloc[0]
            else:
                row = subset.iloc[0]
                
        self.current_step = 0
        
        # State: [Recency, Frequency, Monetary, PC1, PC2]
        self.state = np.array([
            row['Recency'],
            row['Frequency'],
            row['Monetary'],
            row.get('PC1', 0.0),
            row.get('PC2', 0.0)
        ], dtype=np.float32)
        
        return self.state
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.
        Calculates expected reward:
          Reward = Predicted Spend * Probability Adjustment[action] - Cost[action]
        Updates state features based on a random purchase decision:
          If purchase: Recency = 0, Frequency += 1, Monetary += Predicted Spend
          If no purchase: Recency += 30, Frequency/Monetary constant.
        """
        self.current_step += 1
        
        # Predict spend using the regression model
        # Input features: [Recency, Frequency, Monetary, PC1, PC2]
        pred_spend = self.regressor.predict(self.state.reshape(1, -1))[0]
        pred_spend = max(0.0, float(pred_spend))
        
        # Calculate Reward based on requirements formula
        prob = self.prob_adj[action]
        cost = self.costs[action]
        reward = pred_spend * prob - cost
        
        # Simulate transition dynamics
        purchased = np.random.rand() < prob
        
        recency, frequency, monetary, pc1, pc2 = self.state
        if purchased:
            recency = 0.0
            frequency += 1.0
            monetary += pred_spend
        else:
            recency += 30.0  # Increment recency by 30 days
            
        self.state = np.array([recency, frequency, monetary, pc1, pc2], dtype=np.float32)
        done = self.current_step >= self.max_steps
        
        return self.state, reward, done, {'purchased': purchased, 'spend': pred_spend if purchased else 0.0}

# ==========================================
# TABULAR Q-LEARNING
# ==========================================
class TabularQLearningAgent:
    """
    Q-Learning Agent using KMeans to discretize continuous states.
    Q-table shape: 8 states x 3 actions
    """
    def __init__(self, n_clusters: int = 8, action_dim: int = 3, lr: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        self.n_clusters = n_clusters
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_clusters, action_dim), dtype=np.float32)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit_kmeans(self, states: np.ndarray) -> None:
        """
        Fits the KMeans discretizer on the customer state space.
        """
        self.kmeans.fit(states.astype(np.float32))
        
    def discretize(self, state: np.ndarray) -> int:
        """
        Maps a continuous state vector to a discrete cluster index.
        """
        return int(self.kmeans.predict(state.reshape(1, -1).astype(np.float32))[0])
        
    def choose_action(self, state_idx: int, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        """
        if not eval_mode and np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return int(np.argmax(self.q_table[state_idx]))
        
    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        """
        Q-table update rule.
        """
        best_next = np.max(self.q_table[next_state_idx]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        self.q_table[state_idx, action] += self.lr * (td_target - self.q_table[state_idx, action])

# ==========================================
# DEEP Q NETWORK (DQN)
# ==========================================
class DQNNet(nn.Module):
    """
    PyTorch neural network for Q-value approximation.
    Architecture: Input -> 64 ReLU -> 64 ReLU -> Output 3
    """
    def __init__(self, state_dim: int = 5, action_dim: int = 3):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    """
    Replay buffer to store transitions for DQN experience replay.
    """
    def __init__(self, capacity: int = 2000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent with Target Network and Experience Replay.
    """
    def __init__(self, state_dim: int = 5, action_dim: int = 3, lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, decay_steps: int = 2000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = (epsilon_start - epsilon_end) / decay_steps
        
        self.policy_net = DQNNet(state_dim, action_dim)
        self.target_net = DQNNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=2500)
        self.batch_size = 64
        
    def choose_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())
            
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.decay_rate)
        
    def update(self) -> Optional[float]:
        """
        Performs one gradient step using a sample from the replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Get Q values for current states
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute max Q values for next states using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self) -> None:
        """
        Copies policy network weights to target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, filepath: str) -> None:
        """
        Saves the policy network state dictionary.
        """
        torch.save(self.policy_net.state_dict(), filepath)
        
    def load(self, filepath: str) -> None:
        """
        Loads the policy network state dictionary.
        """
        self.policy_net.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()

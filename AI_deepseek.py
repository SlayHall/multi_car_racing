import gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gym_multi_car_racing

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
LEARNING_RATE = 0.001
NUM_EPISODES = 1000
BUFFER_CAPACITY = 100000
NUM_AGENTS = 2

# Discretized actions (steering, gas, brake)
DISCRETE_ACTIONS = [
    [-1, 0, 0],  # Turn left
    [1, 0, 0],   # Turn right
    [0, 1, 0],   # Accelerate
    [0, 0, 1],   # Brake
    [0, 0, 0]    # No-op
]

def process_frame(frame):
    """Convert frame to grayscale and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray / 255.0  # Normalize to [0, 1]

class ReplayBuffer:
    """Stores transitions and samples batches for training."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32))
        dones = torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for processing stacked frames."""
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_input_size = self._get_conv_output((input_channels, 96, 96))
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers."""
        x = torch.zeros(1, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize environment
env = gym.make('MultiCarRacing-v0', num_agents=NUM_AGENTS)

# Initialize DQNs, target networks, optimizers, and replay buffers for each agent
agents_dqn = []
agents_target_dqn = []
agents_optimizer = []
agents_buffer = []

for _ in range(NUM_AGENTS):
    dqn = DQN(input_channels=4, num_actions=5)
    target_dqn = DQN(input_channels=4, num_actions=5)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    agents_dqn.append(dqn)
    agents_target_dqn.append(target_dqn)
    agents_optimizer.append(optimizer)
    agents_buffer.append(ReplayBuffer(BUFFER_CAPACITY))

epsilon = EPS_START

# Training loop
for episode in range(NUM_EPISODES):
    observations = env.reset()
    frame_buffers = [deque(maxlen=4) for _ in range(NUM_AGENTS)]
    
    # Initialize frame buffers with the first frame repeated 4 times
    for i in range(NUM_AGENTS):
        processed = process_frame(observations[i])
        for _ in range(4):
            frame_buffers[i].append(processed)
    
    done = False
    total_rewards = [0.0] * NUM_AGENTS
    
    while not done:
        current_states = []
        for i in range(NUM_AGENTS):
            state = np.array(frame_buffers[i], dtype=np.float32)
            current_states.append(state)
        
        # Select actions using epsilon-greedy policy
        actions = []
        for i in range(NUM_AGENTS):
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                state_tensor = torch.FloatTensor(current_states[i]).unsqueeze(0)
                with torch.no_grad():
                    q_values = agents_dqn[i](state_tensor)
                action = q_values.argmax().item()
            actions.append(DISCRETE_ACTIONS[action])
        
        # Take action in the environment
        next_observations, rewards, dones, _ = env.step(actions)
        # done = all(dones)
        
        # Process next observations and update frame buffers
        next_states = []
        for i in range(NUM_AGENTS):
            processed_next = process_frame(next_observations[i])
            frame_buffers[i].append(processed_next)
            next_state = np.array(frame_buffers[i], dtype=np.float32)
            next_states.append(next_state)
        
        # Store transitions in replay buffers
        for i in range(NUM_AGENTS):
            agents_buffer[i].add(current_states[i], actions[i], rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
        
        # Train each agent if enough samples are available
        for i in range(NUM_AGENTS):
            if len(agents_buffer[i]) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = agents_buffer[i].sample(BATCH_SIZE)
                
                # Compute current Q values
                current_q = agents_dqn[i](states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values
                with torch.no_grad():
                    next_q = agents_target_dqn[i](next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * GAMMA * next_q
                
                # Compute loss and optimize
                loss = F.mse_loss(current_q, target_q)
                agents_optimizer[i].zero_grad()
                loss.backward()
                agents_optimizer[i].step()
        
        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            for i in range(NUM_AGENTS):
                agents_target_dqn[i].load_state_dict(agents_dqn[i].state_dict())
        
        # Decay epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    print(f"Episode {episode + 1}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.2f}")

env.close()

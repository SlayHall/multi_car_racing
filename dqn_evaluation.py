import gym
import gym_multi_car_racing
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time

# -----------------------------
# Define the DQN model
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_shape=(4, 96, 96), num_actions=5):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32,
                               kernel_size=5, stride=2, padding=1)  # (4,96,96) -> (32,48,48)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=2, padding=1)  # (32,48,48) -> (64,24,24)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=2, padding=1)  # (64,24,24) -> (64,12,12)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=1)  # (64,12,12) -> (128,6,6)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # Flattened size: 4608 -> 512
        self.fc2 = nn.Linear(512, num_actions)  # 512 -> 5 (Q-values)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Set up device and load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN().to(device)
# Load the model state dictionary (ensure the file "dqn_model.pth" exists)
dqn.load_state_dict(torch.load("dqn_model_300.pth", map_location=device))
dqn.eval()  # Set model to evaluation mode

# -----------------------------
# Environment and Preprocessing
# -----------------------------
env = gym.make("MultiCarRacing-v0",
               num_agents=2,
               direction='CCW',
               use_random_direction=True,
               backwards_flag=True,
               h_ratio=0.25,
               use_ego_color=False)

# Parameters for frame stacking
buffer_size = 4  # Number of frames to stack

# Initialize frame buffers for each agent as deques
frame_buffers = [deque(maxlen=buffer_size) for _ in range(2)]
# Pre-fill frame buffers with zeros (each frame is 96x96)
for i in range(2):
    for _ in range(buffer_size):
        frame_buffers[i].append(np.zeros((96, 96), dtype=np.float32))

def grayscale(obs):
    """
    Convert a list of RGB images to grayscale and normalize them.
    Assumes obs is a list of images, one per agent.
    """
    gray_obs = [cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY) for ob in obs]
    normalized_obs = [gray / 255.0 for gray in gray_obs]
    return normalized_obs  # returns a list of 2 arrays of shape (96,96)

def update_frame_buffers(new_frames):
    """
    Append the new frames (one per agent) to the corresponding frame buffers.
    """
    for i in range(2):
        frame_buffers[i].append(new_frames[i])

def get_state_tensor():
    """
    Convert the current frame buffers into a PyTorch tensor.
    Returns a tensor of shape (num_agents, buffer_size, 96, 96).
    """
    # Convert each agent's frame buffer (a deque) into a numpy array
    stacked_frames = [np.stack(list(frame_buffers[i]), axis=0) for i in range(2)]
    # Create a numpy array with shape (2, buffer_size, 96, 96)
    state_np = np.array(stacked_frames, dtype=np.float32)
    # Convert to tensor and add a batch dimension (resulting shape: (1, 2, buffer_size, 96, 96))
    state_tensor = torch.from_numpy(state_np).to(device)
    return state_tensor

def map_action(action_index):
    """
    Map a discrete action index to the corresponding continuous action.
    This mapping should match the one used during training.
    """
    mapping = {
        0: [-1, 0, 0],  # Left
        1: [1, 0, 0],   # Right
        2: [0, 1, 0],   # Accelerate
        3: [0, 0, 1],   # Brake
        4: [0, 0, 0]    # Do nothing
    }
    return mapping.get(action_index, [0, 0, 0])

# -----------------------------
# Run the Model in the Environment
# -----------------------------
obs = env.reset()
# Preprocess the initial observation
gray_obs = grayscale(obs)
update_frame_buffers(gray_obs)

done = False
total_reward = 0
while not done:
    state_tensor = get_state_tensor()  # Shape: (2, buffer_size, 96, 96)
    actions = []
    with torch.no_grad():
        # For each agent, select the action with highest Q-value
        for i in range(2):
            # Extract the i-th agent's state and add a batch dimension: shape becomes (1, buffer_size, 96, 96)
            agent_state = state_tensor[i].unsqueeze(0)
            q_values = dqn(agent_state)
            action_index = torch.argmax(q_values).item()
            actions.append(map_action(action_index))
    # Step the environment with the actions for each agent
    obs, reward, done, info = env.step(actions)
    total_reward += reward

    # Preprocess the new observation and update frame buffers
    gray_obs = grayscale(obs)
    update_frame_buffers(gray_obs)

    env.render()  # Render the environment (optional)

print("Total reward:", total_reward)
env.close()

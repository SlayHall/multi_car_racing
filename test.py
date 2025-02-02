import gym
import gym_multi_car_racing
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


########################## Step 1: Initialize the environment --------------------------------------------

# Initialize the environment----------------------------------------------------------------------------
env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)


########################## Step 2 Preprocess Observations (Grayscale)------------------------------------------------------------

# Create lists for storing frames for each agent--------------------------------------------------------------------------------
frame_buffers = [[], []]  # Two agents, so two lists  shape(2, 4, 96, 96)
buffer_size = 4           # Number of frames to stack
buffer_skip = 4           # Number of steps to skip for faster processing
'''
#!  print ("frame_buffers:", {np.array(frame_buffers).shape})
'''
for i in range(2):
    for _ in range(4):
        frame_buffers[i].append(np.zeros((96, 96)))  # Initialize the buffer with zeros

def buffer_append(new_frame):
  for i in range(2):  
        frame_buffers[i].append(new_frame[i])  # Store the respective agent's frame
        if len(frame_buffers[i]) > buffer_size:
            frame_buffers[i].pop()


def grayscale(frame):
  gray_obs = [cv2.cvtColor(obs[i], cv2.COLOR_RGB2GRAY) for i in range(env.num_agents)]
  normalized_obs = [gray_obs[i] / 255.0 for i in range(env.num_agents)]
  return np.array(normalized_obs)    #shape(2, 96, 96)
  '''
  #! print("normalized_obs:", {np.array(normalized_obs).shape})
  #! print("frame:", {np.array(frame).shape})
  #! print("gray_obs:", {np.array(gray_obs).shape})
  '''
  return np.array(gray_obs)          #shape(2, 96, 96)

########################### step 3: Build the Deep Q-Network (DQN)------------------------------------------------------------

def deiscrete_action_space(action):
   switch_dict = {    # *Discrete*    *Q-values*
      0: [-1,0,0],    # Left            1.2    
      1: [1,0,0],     # Right          -0.5
      2: [0,1,0],     # Accelerate      0.8
      3: [0,0,1],     # Brake           2.1
      4: [0,0,0]      # Do nothing      0.3
   }
   return switch_dict.get(action, "Invalid action not (0-4)")

class DQN(nn.Module):
  def __init__(self, input_shape=(4, 96, 96), num_actions=5):
    super(DQN, self).__init__()
                                                                                                             # *input_shape* 
    self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, stride=2, padding=1)  # (4, 96, 96) -> (32, 48, 48)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)              # (32, 48, 48) -> (64, 24, 24)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)              # (64, 24, 24) -> (64, 12, 12)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)             # (64, 12, 12) -> (128, 6, 6)

    self.fc1 = nn.Linear(128 * 6 * 6, 512)      #input 4608 output 512 flatten the output of the last conv layer
    self.fc2 = nn.Linear(512, num_actions)     #input 512 output 5  q-values for each action

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(x.size(0), -1)     # Flatten the output of the last conv layer
    x = F.relu(self.fc1(x))
    x = self.fc2(x)               # Output the Q-values for each action
    return x


# Reset the environment---------------------------------------------------------------------------------
obs = env.reset()
done = False
total_reward = 0

# Start the simulation---------------------------------------------------------------------------------
while not done:
  
  action = [[np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)], [np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)]]

  
  # obs is of shape (num_agents, 96, 96, 3)
  # reward is of shape (num_agents,)
  # done is a bool and info is not used (an empty dict).
  obs, reward, done, info = env.step(action)

  # Process the first observation and fill the buffers
  grayscale_obs = grayscale(obs)
  buffer_append(grayscale_obs)
  
  total_reward += reward
  env.render()

print("individual scores:", total_reward)
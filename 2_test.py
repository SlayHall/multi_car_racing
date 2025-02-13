import gym
import gym_multi_car_racing
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

import time
import sys
from playsound import playsound

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self, message):
        """Stop the timer, print the elapsed time, and return it."""
        if self.start_time is None:
            print("Error: Timer was not started!")
            return None
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        print(f"{message}:  {elapsed_time:.6f} seconds")
        return elapsed_time

timer = Timer()

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

for i in range(2):
    for _ in range(4):
        frame_buffers[i].append(np.zeros((96, 96)))  # Initialize the buffer with zeros

def buffer_append(new_frame):
    for i in range(2):
        frame_buffers[i].append(new_frame[i])  # Store the respective agent's frame
        if len(frame_buffers[i]) > buffer_size:
            frame_buffers[i].pop(0)

def grayscale(frame):
    gray_obs = [cv2.cvtColor(frame[i], cv2.COLOR_RGB2GRAY) for i in range(env.num_agents)]
    normalized_obs = [gray_obs[i] / 255.0 for i in range(env.num_agents)]
    return np.array(normalized_obs)    # shape(2, 96, 96)

########################### Step 3: Build the Deep Q-Network (DQN)------------------------------------------------------------

def deiscrete_action_space(action):
    switch_dict = {    # *Discrete*    *Q-values*
        0: [-1, 0, 0],    # Left            1.2    
        1: [1, 0, 0],     # Right          -0.5
        2: [0, 1, 0],     # Accelerate      0.8
        3: [0, 0, 1],     # Brake           2.1
        4: [0, 0, 0]      # Do nothing      0.3
    }
    return switch_dict.get(action, "Invalid action not (0-4)")

class DQN(nn.Module):
    def __init__(self, input_shape=(4, 96, 96), num_actions=5):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, stride=2, padding=1)  # (4, 96, 96) -> (32, 48, 48)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)              # (32, 48, 48) -> (64, 24, 24)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)              # (64, 24, 24) -> (64, 12, 12)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)             # (64, 12, 12) -> (128, 6, 6)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)     # input 4608 output 512 flatten the output of the last conv layer
        self.fc2 = nn.Linear(512, num_actions)     # input 512 output 5  q-values for each action

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)     # Flatten the output of the last conv layer
        x = F.relu(self.fc1(x))       # Pass through the fully connected layer
        x = self.fc2(x)               # Output the Q-values for each action
        return x

# select computing device and number of episodes-----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

dqn = DQN().to(device)      # Initialize the DQN model

def Convert_Frame_Buffer_to_Tensor(frame_buffers):
    stacked_frames = []
    for i in range(2):
        frames = frame_buffers[i][-buffer_size:]                # Get the last 4 frames for each agent
        stacked_frames.append(np.stack(frames, axis=0))         # Stack them along a new dimension
    stacked_frames = np.array(stacked_frames)                   # Convert the list of arrays into a single numpy array  
    return torch.from_numpy(stacked_frames).float()  # More efficient conversion
    return torch.tensor(stacked_frames, dtype=torch.float32)    # Convert the stacked frames to a tensor

########################### Step 4: Implement the Replay Buffer------------------------------------------------------------

# 1) Create a ReplayBuffer class.
class ReplayBuffer:
    def __init__(self):
        self.rreplya_buffer_deque = deque(maxlen=20000)  # Initialize the buffer with a maximum length of 100000

    def add(self, state, action, reward, next_state, done):

        #state = state.cpu().detach().numpy()  # Convert the state tensor to a numpy array
        #next_state = next_state.cpu().detach().numpy()  # Convert the next state tensor to a numpy array
        #action = action.cpu().detach().numpy()  # Convert the action tensor to a numpy array
        #reward = reward.cpu().detach().numpy()  # Convert the reward tensor to a numpy array
        #done = done.cpu().detach().numpy()  # Convert the done tensor to a numpy array


        self.rreplya_buffer_deque.append((state, action, reward, next_state, done))  # Add the transition to the buffer

    def sample(self, sample_batch_size):
        sample = random.sample(self.rreplya_buffer_deque, sample_batch_size)          # Sample a batch of transitions without replacements, they are tuples that must be unpacked
        state, action, reward, next_state, done = map(list, zip(*sample))      # Unpack the batch of transitions
        
        ## Convert the batch of transitions to tensors--------------------------------
        state = torch.stack(state).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.stack(next_state).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)
        

        return state, action, reward, next_state, done

# 2) Initialize a separate replay buffer for each agent.
replay_buffer_list = [ReplayBuffer(), ReplayBuffer()]  # Initialize the replay buffer for 2 agents  

########################## Step 5: Train the DQN---------------------------------------------------------------------------------

# Create a target network and initialize its weights to match dqn.
target_dqn = DQN().to(device)
target_dqn.load_state_dict(dqn.state_dict())    # Initialize the target network with the same weights as the DQN
target_dqn.eval()               # Set target network to evaluation mode
target_update_frequency = 10    # Update the target network every 10 episodes


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

aadam_learning_rate = 0.001
optimizer = torch.optim.Adam(dqn.parameters(), lr=aadam_learning_rate)
gamma = 0.99

batch_size = 32

render_every = 50

num_episodes = 600

episode_durations = []

for i_episode in range(num_episodes):              # Initialize the episode
    
    start_episode = time.perf_counter()
    
    obs = env.reset()
    done = False
    total_reward = 0

    for agent in range(2):
        frame_buffers[agent] = [np.zeros((96, 96)) for _ in range(buffer_size)]  # Initialize the buffer with zeros

    grayscale_obs = grayscale(obs)
    buffer_append(grayscale_obs)
    loops = 0

    while not done:                                  # Start training episode

        loops += 1
        print(f"loops completed: {loops} and ", end="\r", flush=True)
        #deque length: {len(replay_buffer_list[agent].rreplya_buffer_deque)} of size {sys.getsizeof(replay_buffer_list[agent].rreplya_buffer_deque)/(1024*1024)} MB and torch memory: {(torch.cuda.memory_allocated()/(1024*1024))} MB
        
        state_tensor = Convert_Frame_Buffer_to_Tensor(frame_buffers).to(device)  # shape(1, 2, 4, 96, 96) 1 batch, 2 agents, 4 frames, 96x96 pixels move to device

        q_actions = []
        
        # 1) Use the epsilon-greedy policy to select an action for each agent.
        for agent in range(2):
            if random.random() < epsilon:                                    # Random action exploration
                q_actions.append(np.random.randint(5))          
            else:
                with torch.no_grad():                                        # Exploitation
                    q_values = dqn(state_tensor[agent].unsqueeze(0))
                    q_actions.append(torch.argmax(q_values).item())
        
        descreat_actions = [deiscrete_action_space(a) for a in q_actions]    # Convert the action to the discrete action space 

        obs, reward, done, info = env.step(descreat_actions)                 # Step the environment, obs is of shape (num_agents, 96, 96, 3), reward is of shape (num_agents,)
        
        grayscale_obs = grayscale(obs)               # Process the observation and fill the buffers
        buffer_append(grayscale_obs)
    
        total_reward += reward

        next_state_tensor = Convert_Frame_Buffer_to_Tensor(frame_buffers).to(device)

        # 2) Store the transition in the replay buffer.
        
        for agent in range(2):
            replay_buffer_list[agent].add(state_tensor[agent].detach().cpu(), q_actions[agent], reward[agent], next_state_tensor[agent].detach().cpu(), done)
            
      
        # 3) Sample a batch of transitions from the replay buffer and calculate the loss.
        if len(replay_buffer_list[0].rreplya_buffer_deque) > batch_size:
            for agent in range(2):
                state, action, reward, next_state, r_done = replay_buffer_list[agent].sample(batch_size)
                
                q_values = dqn(state)               # Get the Q-values for the current state
                

                with torch.no_grad():
                    next_q_values = target_dqn(next_state)     # Get the Q-values for the next state
                target_q_values = q_values.clone()  # Clone the Q-values to calculate the target Q-values
                
                for i in range(batch_size):
                    target_q_values[i, action[i]] = reward[i] + gamma * torch.max(next_q_values[i]) * (1 - r_done[i])     # Calculate the target Q-values using the Bellman equation

                loss = F.mse_loss(q_values, target_q_values)    # Calculate the loss using the mean squared error loss function   
                

                optimizer.zero_grad()   # Zero the gradients
                loss.backward()         # Backpropagate the loss
                optimizer.step()        # Update the DQN parameters

    # end of episode  --------------------------------------------------------------------------------
    epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Decay the epsilon value
    torch.cuda.empty_cache()                             # Clear the cache to prevent memory leaks

    episode_duration = time.perf_counter() - start_episode
    episode_durations.append(episode_duration)
    avg_duration = sum(episode_durations) / len(episode_durations)
    remaining_episodes = num_episodes - i_episode - 1
    expected_remaining_time = avg_duration * remaining_episodes
    remaining_minutes = expected_remaining_time / 60.0
    if remaining_minutes > 59:
        hours = int(remaining_minutes // 60)
        minutes = int(remaining_minutes % 60)
        expected_str = f"{hours}h {minutes}m"
    else:
        expected_str = f"{remaining_minutes:.2f} minutes"


    print("individual scores:", total_reward)
    print("epsilon:", epsilon)
    #print("episode:", i_episode)
    print("loss:", loss.item())
    print(f"Episode {i_episode+1} finished in {episode_duration:.2f} seconds. "
          f"Expected time to finish: {expected_str}.")
    print("------------------------------------------------------")

    if i_episode % target_update_frequency == 0:  # Update the target network every 10 episodes
        target_dqn.load_state_dict(dqn.state_dict())
        print("Target network updated!")
        print("-----------------------###----------------------------")


    
#playsound('mixkit-bell-notification-933.wav')
########################## Step 6: Evaluate the DQN---------------------------------------------------------------------------------
print("Training complete!")
see = input("Do you wanna render the environment? (y/n): ")
if see.lower() == 'y':
    # Optionally, set the model to evaluation mode.
    dqn.eval()
    
    obs = env.reset()
    done = False
    # Reset or initialize the frame buffers appropriately
    for agent in range(2):
        frame_buffers[agent] = [np.zeros((96, 96)) for _ in range(buffer_size)]
    grayscale_obs = grayscale(obs)
    buffer_append(grayscale_obs)
    
    while not done:
        env.render()
        
        # Update state from frame buffers
        state_tensor = Convert_Frame_Buffer_to_Tensor(frame_buffers).to(device)
        q_actions = []
        with torch.no_grad():
            for agent in range(2):
                q_values = dqn(state_tensor[agent].unsqueeze(0))
                q_actions.append(torch.argmax(q_values).item())
                print(f"Agent {agent} Q-values: {q_values.cpu().numpy()} selected action: {q_actions[agent]}")
        
        discrete_actions = [deiscrete_action_space(a) for a in q_actions]
        obs, reward, done, info = env.step(discrete_actions)
        
        # Update the frame buffers with the new observation
        grayscale_obs = grayscale(obs)
        buffer_append(grayscale_obs)
        
        time.sleep(0.05)  # Add a small delay so that rendering is visible
        
else:
    print("Ok, just watch")
    env.render()
    time.sleep(5)
    







rspounce = input("Do you want to save the model? (y/n): ")
if rspounce == 'y':
    torch.save(dqn.state_dict(), "dqn_model_600.pth")
    print("Model saved successfully!")
elif rspounce == 'n':
    print("Model not saved!")
else:
    torch.save(dqn.state_dict(), "dqn_model_600.pth")
    print("Model saved successfully! anyway lol")

env.close()  # Close the environment after training
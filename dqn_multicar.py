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
import torch.optim as optim
import matplotlib.pyplot as plt





########################## helping code for debuging ------------------------------------------------------------###

'''-------------------------time calculating class---------------------------------------------------------------'''
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

timer = Timer()    # Create a timer object


'''---------------------------------wights visulization--------------------------------------------------------'''
plt.ion()  # Enable interactive mode for matplotlib

def update_all_conv_weights(agent, axes):
    layers = [agent.conv1, agent.conv2, agent.conv3, agent.conv4]          # List of all convolutional layers
    layer_names = ["Conv1", "Conv2", "Conv3", "Conv4"]                     # Names of the layers

    for ax in axes:                                                        # Clear previous contents of the axes
        ax.clear()

    for layer_idx, (layer, name) in enumerate(zip(layers, layer_names)):   # Plot weights for each layer
        weights = layer.weight.data.cpu().numpy()
        num_filters = weights.shape[0]

        #                                                                  # Plot filters for the current layer
        for filter_idx in range(min(num_filters, 8)):                      # Show up to 8 filters per layer
            filter_weights = np.mean(weights[filter_idx, :, :, :], axis=0) # Average across input channels
            ax = axes[layer_idx * 8 + filter_idx]                          # Calculate the subplot index
            ax.imshow(filter_weights, cmap='gray')
            ax.set_title(f"{name} Filter {filter_idx}")
            ax.axis('off')

        for filter_idx in range(num_filters, 8):                           # Hide unused subplots for the current layer
            ax = axes[layer_idx * 8 + filter_idx]
            ax.axis('off')

    # Update the figure
    plt.tight_layout()
    plt.draw()          # Update the plot without blocking
    plt.pause(0.01)     # Allow time for the plot to update



########################## Step 1: Initialize the environment -----------------------------------###

env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
               use_random_direction=True, backwards_flag=True, h_ratio=0.25,
               use_ego_color=False)


########################## Step 2 Preprocess Observations (Grayscale)----------------------------###

'''------------------------Create lists for storing frames for each agent------------------------'''
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

'''------------------------Convert RGB to Grayscale and Normalize---------------------------------'''
def grayscale(frame):
    gray_obs = [cv2.cvtColor(frame[i], cv2.COLOR_RGB2GRAY) for i in range(env.num_agents)]
    normalized_obs = [(gray_obs[i] / 255.0) for i in range(env.num_agents)]  # Range [0, 1]
    return np.array(normalized_obs)                                          # shape(2, 96, 96)

'''------------------------Normalize the reward---------------------------------------------------'''
def normalize_reward(reward, min_reward = -100, max_reward = 100):           # Normalize to [0, 1]   
    normalized_reward = [(r - min_reward) / (max_reward - min_reward) for r in reward]
    return normalized_reward


########################### Step 3: Build the Deep Q-Network (DQN)--------------------------------------------------------------------###

'''------------------------Define the Discrete Action Space function------------------------------------------------------------------'''
def deiscrete_action_space(action):
    switch_dict = {    # *Discrete*    *Q-values*
        0: [-1, 0, 0],    # Left            1.2    
        1: [1, 0, 0],     # Right          -0.5
        2: [0, 1, 0],     # Accelerate      0.8
        3: [0, 0, 1],     # Brake           2.1
        4: [0, 0, 0]      # Do nothing      0.3
    }
    return switch_dict.get(action, "Invalid action not (0-4)")  # Return the action from the dictionary or an error message just in case


'''------------------------Define the DQN model---------------------------------------------------------------------------------------'''
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


'''------------------------Define the Enhanced DQN model---------enhanced model but twice as slow as the above model------------------'''
# class DQN(nn.Module):              # enhanced model but twice as slow as the above model
#     def __init__(self, input_shape=(4, 96, 96), num_actions=5):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (32, 48, 48)
        
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (64, 24, 24)
        
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (64, 12, 12)
        
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (128, 6, 6)

#         self.fc1 = nn.Linear(128 * 6 * 6, 512)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, num_actions)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)
        
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)
        
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool3(x)
        
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool4(x)
        
        
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

'''------------------------Convert the frame buffers to a tensor--------so the input of the DQN becomes a tensor---------------------'''

def Convert_Frame_Buffer_to_Tensor(frame_buffers):
    stacked_frames = [ ]
    for i in range(2):
        frames = frame_buffers[i][-buffer_size:]                # Get the last 4 frames for each agent
        stacked_frames.append(np.stack(frames, axis=0))         # Stack them along a new dimension stacked frames (4, 96, 96)
    stacked_frames = np.array(stacked_frames)                   # Convert the list of arrays into a single numpy array  
    return torch.from_numpy(stacked_frames).float().to(device)  # More efficient conversion
    return torch.tensor(stacked_frames, dtype=torch.float32)    # Convert the stacked frames to a tensor (not used in this implementation)


########################### Step 4: Implement the Replay Buffer------------------------------------------------------------###

'''--------------------1) Create a ReplayBuffer class----------------------------------------------------------------------'''
class ReplayBuffer:
    def __init__(self):
        self.rreplya_buffer_deque = deque(maxlen=18000)  # Initialize the buffer with a maximum length of 18000 

    def add(self, state, action, reward, next_state, done):
        self.rreplya_buffer_deque.append((state, action, reward, next_state, done))   # Add the transition to the buffer

    def sample(self, sample_batch_size):
        sample = random.sample(self.rreplya_buffer_deque, sample_batch_size)          # Sample a batch of transitions without replacements, they are tuples that must be unpacked
        state, action, reward, next_state, done = map(list, zip(*sample))             # Unpack the batch of transitions
        
        '''--Convert the batch of transitions to tensors--'''
        state = torch.stack(state).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.stack(next_state).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)
        
        return state, action, reward, next_state, done

'''---------2) Initialize a separate replay buffer for each agent-----------------------------------------------------------'''
replay_buffer_list = [ReplayBuffer(), ReplayBuffer()]  # Initialize the replay buffer for 2 agents  


########################## Step 5: Train the DQN----------------------------------------------------------###


'''-------------select computing device and number of episodes--------------------------------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50


'''--------------Initialize DQN models for each agent------------------------------------------------------'''
dqn_agents = [DQN().to(device), DQN().to(device)]
target_dqns = [DQN().to(device), DQN().to(device)]

for i in range(2):
    target_dqns[i].load_state_dict(dqn_agents[i].state_dict())  
    target_dqns[i].eval()  # Set the target network to evaluation mode


'''------------------------------ debuging gradiants-------------------------------------------------------'''
gradient_norms = {name: [] for name, _ in dqn_agents[0].named_parameters()}   #Initialize a dictionary to store gradient norms

fig, ax = plt.subplots(figsize=(12, 8))                                       # Initialize the plot for gradient norms
ax.set_xlabel("Training Step")
ax.set_ylabel("Gradient Norm")
ax.set_title("Gradient Norm Over Time")
lines = {name: ax.plot([], [], label=name)[0] for name in gradient_norms.keys()}
ax.legend()


'''------------------------------- wights visualization----------------------------------------------------'''
# fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # Initialize a single figure with subplots for all 4 layers of the first agent 4 rows (one for each layer), 8 columns (for filters)
# fig.suptitle("Agent 0 Conv Layers Weights", fontsize=16)

# axes = axes.flatten()                            # Flatten the axes for easier indexing




'''--------------------------------hyper parameters -------------------------------------------------------'''
target_update_frequency = 10    # Update the target network every 10 episodes

epsilon = 1.0
epsilon_decay = 0.996
epsilon_min = 0.05

aadam_learning_rate = 0.0001
optimizers = [torch.optim.Adam(dqn_agents[i].parameters(), lr=aadam_learning_rate) for i in range(2)]  # Initialize the Adam optimizer for each agent

gamma = 0.99

batch_size = 32

render_every = 50

episode_durations = []

save_file_as = f"dqn_model_{num_episodes}_{epsilon_decay}_{aadam_learning_rate}_{batch_size}.pth"

# dqn.load_state_dict(torch.load("dqn_model_600.pth", map_location=device)) 
# dqn.train() 


''''---------------------------------Training Loop-------------------------------------------------------'''
'''
1-Episode begins 
'''

for i_episode in range(num_episodes):              # Initialize the episode
    
    start_episode = time.perf_counter()
    
    obs = env.reset()
    done = False
    total_reward = 0


    grayscale_obs = grayscale(obs)
    buffer_append(grayscale_obs)

    # for agent in range(2):
    #     frame_buffers[agent] = [np.zeros((96, 96)) for _ in range(buffer_size)]  # Initialize the buffer with zeros

    for i in range(2):
        for _ in range(buffer_size):
            frame_buffers[i].append(grayscale_obs[i])# Initialize the buffer with the first observed frame

    
    loops = 0

    while not done:                                  # Start training episode

        print(f"loops completed: {loops} ", end="\r", flush=True)
        #deque length: {len(replay_buffer_list[agent].rreplya_buffer_deque)} of size {sys.getsizeof(replay_buffer_list[agent].rreplya_buffer_deque)/(1024*1024)} MB and torch memory: {(torch.cuda.memory_allocated()/(1024*1024))} MB
        
        if loops % buffer_skip == 0:                 # process every 4th frame
            grayscale_obs = grayscale(obs)           # Process the observation and fill the buffers
            buffer_append(grayscale_obs)
        
        state_tensor = Convert_Frame_Buffer_to_Tensor(frame_buffers).to(device)  # shape(1, 2, 4, 96, 96) 1 batch, 2 agents, 4 frames, 96x96 pixels move to device 

            
        '''
        2-Perform and action from state and observe the next stat and reward .
        '''
        q_actions = []
        
        # 1) Use the epsilon-greedy policy to select an action for each agent.
        for agent in range(2):
            if random.random() < epsilon:                                    # Random action exploration
                q_actions.append(np.random.randint(5))          
            else:
                with torch.no_grad():                                        # Exploitation
                    q_values = dqn_agents[agent](state_tensor[agent].unsqueeze(0))
                    q_actions.append(torch.argmax(q_values).item())
        
        descreat_actions = [deiscrete_action_space(a) for a in q_actions]    # Convert the action to the discrete action space

        obs, reward, done, info = env.step(descreat_actions)                 # Step the environment, obs is of shape (num_agents, 96, 96, 3), reward is of shape (num_agents,)
        
        
        '''
        3 compute the reward
        '''
        total_reward += reward
        normal_reward = normalize_reward(reward)     # Normalize the reward to [-1, 1]

        '''
        4-next state is the new current state
        '''
        grayscale_obs = grayscale(obs)               # Process the observation and fill the buffers
        buffer_append(grayscale_obs)
        next_state_tensor = Convert_Frame_Buffer_to_Tensor(frame_buffers).to(device)  # shape(1, 2, 4, 96, 96) 1 batch, 2 agents, 4 frames, 96x96 pixels move to device 


        '''
        5-store in the replay buffer
        '''
        # 2) Store the transition in the replay buffer.
        for agent in range(2):
            replay_buffer_list[agent].add(state_tensor[agent].detach().cpu(), q_actions[agent], normal_reward[agent], next_state_tensor[agent].detach().cpu(), done)
            

        '''
        6-run the batch training 
        '''
        # 3) Sample a batch of transitions from the replay buffer and calculate the loss.
        for agent in range(2):
            if len(replay_buffer_list[agent].rreplya_buffer_deque) > batch_size:
                state, action, reward, next_state, r_done = replay_buffer_list[agent].sample(batch_size)

                '''
                Compute Q-values for current state
                '''
                q_values = dqn_agents[agent](state)                # Get the Q-values for the current state
                

                '''
                Compute target Q-values
                '''
                with torch.no_grad():
                    next_q_values = target_dqns[agent](next_state) # Get the Q-values for the next state


                target_q_values = q_values.clone()                 # Clone the Q-values to calculate the target Q-values
                
                for i in range(batch_size):
                    target_q_values[i, action[i]] = reward[i] + gamma * torch.max(next_q_values[i]) * (1 - r_done[i])     # Calculate the target Q-values using the Bellman equation

                
                '''
                compute the loss
                '''
                loss = F.mse_loss(q_values, target_q_values)         # Calculate the loss using the mean squared error loss function   
                #loss = F.smooth_l1_loss(q_values, target_q_values)  # Huber loss for more stable training (not used in this implementation)

                optimizers[agent].zero_grad()   # Zero the gradients
                loss.backward()                 # Backpropagate the loss
                torch.nn.utils.clip_grad_norm_(dqn_agents[agent].parameters(), max_norm=1.0)  # Clip the gradients to prevent exploding gradients(not needed for now sense gradients are vanishing)
                
                '''--debuging the gradiant----collect gradient norms (only for key layers) to plot them--'''
                # for name, param in dqn_agents[agent].named_parameters():
                #     if param.grad is not None and "conv1" in name or "conv4" in name or "fc1" in name or "fc2" in name:
                #         grad_norm = param.grad.norm().item()   # Compute the norm of the gradient
                #         gradient_norms[name].append(grad_norm)

                optimizers[agent].step()         # Update the DQN parameters

        loops += 1
   
    '''
    ### end of episode  -------------------------------------------------------------------------------------###
    '''
    torch.cuda.empty_cache()                               # Clear the cache to prevent memory leaks
    epsilon = max(epsilon * epsilon_decay, epsilon_min)    # Decay the epsilon value


    '''------------------------------------plots for debuging----------------------------------------------'''
    
    '''--Visualize weights at the end of each episode--'''
    #update_all_conv_weights(dqn_agents[0], axes)  # Update weights for the first agent only
    
    

    '''--Plot gradient norms at the end of each episode--'''
    # if i_episode % 1 == 0:  # Plot every episode (adjust frequency as needed)
    #     for name, line in lines.items():
    #         if name in gradient_norms:  # Only update lines for layers we're tracking
    #             line.set_xdata(range(len(gradient_norms[name])))
    #             line.set_ydata(gradient_norms[name])
    #     ax.relim()
    #     ax.autoscale_view()
    #     plt.draw()
    #     plt.pause(0.01)  # Allow time for the plot to update

    # # Clear the gradient storage for the next episode
    # for name in gradient_norms.keys():
    #     gradient_norms[name].clear()
    '''----------------------------------------------------------------------------------------------------'''


    # if i_episode % render_every == 0:
    #     env.render()


    '''
    # calculate the time needed to finish
    '''
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


    '''
    # end of episode report
    '''
    print("individual scores:", total_reward)
    print("epsilon:", epsilon)
    #print("episode:", i_episode)
    print("loss:", loss.item())
    print(f"Episode {i_episode} finished in {episode_duration:.2f} seconds. "
          f"Expected time to finish: {expected_str}.")
    print("------------------------------------------------------")
    

    '''
    7-after N iteration, the weights of the prediction network get copied to Target Network
    '''
    if i_episode % target_update_frequency == 0:  # Update the target network every 10 episodes
        for agent in range(2):
            target_dqns[agent].load_state_dict(dqn_agents[agent].state_dict())        
        print(f"Target network updated! Episode {i_episode}")
        print("-----------------------###----------------------------")



########################## Step 7: Evaluate the DQN-------------------------------------------------------------------------------###
print("Training complete!")


'''------------------------Render the environment---------------------------------------------------------------------------------'''
see = input("Do you wanna render the environment? (y/n): ")
if see.lower() == 'n':
    print("as you wish!")
else:
    # Optionally, set the model to evaluation mode.
    for agent in dqn_agents:
        agent.eval()
    
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
                q_values = dqn_agents[agent](state_tensor[agent].unsqueeze(0))
                q_actions.append(torch.argmax(q_values).item())
                #print(f"Agent {agent} Q-values: {q_values.cpu().numpy()} selected action: {q_actions[agent]}" , end="\r", flush=True)

        print(f"Agent 1 selected action: {q_actions[0]} and agent 2 {q_actions[1]}" , end="\r", flush=True)
        
        discrete_actions = [deiscrete_action_space(a) for a in q_actions]
        obs, reward, done, info = env.step(discrete_actions)
        
        # Update the frame buffers with the new observation
        grayscale_obs = grayscale(obs)
        buffer_append(grayscale_obs)
                

'''------------------------Save the model---------------------------------------------------------------------------------'''
rspounce = input("Do you want to save the model? (y/n): ")

if rspounce == 'y':
    for agent in range(2):
        torch.save(dqn_agents[agent].state_dict(), f"agent_{agent}_{save_file_as}")
    print("Model saved successfully!")

elif rspounce == 'n':
    print("Model not saved!")

else:
    for agent in range(2):
        torch.save(dqn_agents[agent].state_dict(), f"agent_{agent}_{save_file_as}")
    print("Model saved successfully! anyway lol")

env.close()  # Close the environment after training
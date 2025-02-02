import gym
import gym_multi_car_racing
import numpy as np
import cv2



# Initialize the environment----------------------------------------------------------------------------
env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

# Reset the environment---------------------------------------------------------------------------------
obs = env.reset()
done = False
total_reward = 0

# Create lists for storing frames for each agent-------------------------------------------------------
frame_buffers = [[], []]  # Two agents, so two lists
buffer_size = 4           # Number of frames to stack
buffer_skip = 4           # Number of steps to skip for faster processing

def buffer_append(buffer = frame_buffers, new_frame = None):
  buffer.append(new_frame)
  if len(buffer) > buffer_size:
    buffer.pop(0)
  return np.stack(buffer, axis=0)



def grayscale(frame):
  
  gray_obs = [cv2.cvtColor(obs[i], cv2.COLOR_RGB2GRAY) for i in range(env.num_agents)]
  #normalized_obs = [gray_obs[i] / 255.0 for i in range(env.num_agents)]
  #return normalized_obs
  return np.array(gray_obs)


# Start the simulation---------------------------------------------------------------------------------
while not done:
  
  action = [[np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)], [np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)]]

  
  # obs is of shape (num_agents, 96, 96, 3)
  # reward is of shape (num_agents,)
  # done is a bool and info is not used (an empty dict).
  obs, reward, done, info = env.step(action)

  # Process the first observation and fill the buffers
  grayscale_obs = grayscale(obs)
  print("grayscale_obs:", {np.array(grayscale_obs).shape})
  buffer_append(frame_buffers, grayscale_obs)

  total_reward += reward
  env.render()

print("individual scores:", total_reward)
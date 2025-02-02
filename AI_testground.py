import cv2
import numpy as np
import gym
import gym_multi_car_racing

# Parameters
FRAME_STACK = 4  # Number of frames to stack
FRAME_SKIP = 4   # Number of steps to skip for faster processing

# Initialize the environment
env = gym.make("MultiCarRacing-v0", num_agents=2)

# Create lists for storing frames for each agent
# Each agent gets a list to store the last 4 frames
frame_buffers = [[], []]  # Two agents, so two lists

def preprocess_frame(frame):
    """
    Simplified function to process a single frame:
    1. Convert to grayscale.
    2. Normalize pixel values to [0, 1].
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    normalized_frame = gray_frame / 255.0  # Normalize pixel values
    return normalized_frame  # The shape is (96, 96)

def stack_frames(buffer, new_frame):
    """
    Add a new frame to the buffer and return a stack of the last 4 frames.
    1. Add the new frame to the buffer.
    2. If there are more than 4 frames, remove the oldest frame.
    3. Stack the last 4 frames.
    """
    buffer.append(new_frame)  # Add the new frame to the buffer

    # Keep only the last 4 frames
    if len(buffer) > FRAME_STACK:
        buffer.pop(0)  # Remove the oldest frame

    # Stack the frames to create a tensor of shape (4, 96, 96)
    return np.stack(buffer, axis=0)

# Reset the environment
obs = env.reset()

# Process the first observation and fill the buffers
for agent_id in range(env.num_agents):
    first_frame = preprocess_frame(obs[agent_id])  # Process the initial frame
    # Fill the buffer with 4 copies of the initial frame
    for _ in range(FRAME_STACK):
        frame_buffers[agent_id].append(first_frame)

# Start the simulation
done = False
while not done:
    # Generate random actions for both agents
    action = [[np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)] for _ in range(env.num_agents)]

    # Step the environment multiple times for frame skipping
    for _ in range(FRAME_SKIP):
        obs, reward, done, info = env.step(action)  # Take a step
        env.render()  # Render the environment
        if done:  # If the race is over, stop
            break

    # Process the new observation for both agents
    new_frames = [preprocess_frame(obs[agent_id]) for agent_id in range(env.num_agents)]

    # Update the frame buffers and stack the last 4 frames
    stacked_obs = []
    for agent_id in range(env.num_agents):
        stacked_obs.append(stack_frames(frame_buffers[agent_id], new_frames[agent_id]))

    # Print the shape of the stacked frames to confirm correctness
    print(f"Stacked frames shape for both agents: {np.array(stacked_obs).shape}")

# Close the environment when done
env.close()

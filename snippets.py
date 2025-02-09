log_path = os.path.join('training', 'logs')

model=PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# select computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50


# init Double DQN models
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(66,66,n_actions,ACTION_SKIP).to(device)
target_net = DQN(66,66,n_actions,ACTION_SKIP).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

# Initialize environment and agent with Experience Replay Buffer
env = gym.make('Breakout-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

# Train the DQN agent with Experience Replay Buffer
batch_size = 32
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay(batch_size)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


    

class DQN(nn.Module):
    def __init__(self, input_shape=(4, 96, 96), num_actions=5):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], 64 * 8 * 8)  # Ensure flattened tensor matches expected input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x   
    

def get_stacked_state():
    stacked_states = [np.stack(frame_buffers[i], axis=0) for i in range(2)]
    return torch.tensor(stacked_states, dtype=torch.float32)

while not done:
    state_tensor = get_stacked_state().unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = dqn(state_tensor)
    
    actions = [torch.argmax(q_values[i]).item() for i in range(2)]  
    action_list = [discrete_action_space(a) for a in actions]
    
    obs, reward, done, info = env.step(action_list)

'''
The warning: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow...

is informing you that converting a list of NumPy arrays to a tensor is less efficient. If you run into performance issues later, consider converting the list to a single NumPy array first. For example, you can modify your function as follows:
'''
def Convert_Frame_Buffer_to_Tensor(frame_buffers):
    stacked_frames = []
    for i in range(2):
        frames = frame_buffers[i][-buffer_size:]
        stacked_frames.append(np.stack(frames, axis=0))
    # Convert the list of arrays into a single numpy array before converting to tensor
    stacked_frames = np.array(stacked_frames)
    return torch.tensor(stacked_frames, dtype=torch.float32)

################# debuging #################
import tracemalloc                          #for memmory usage
tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()
# ... run some iterations of your code ...
snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)

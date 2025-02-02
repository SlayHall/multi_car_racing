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
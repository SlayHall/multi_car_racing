log_path = os.path.join('training', 'logs')

model=PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# select computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# init Double DQN models
policy_net = DQN(66,66,n_actions,ACTION_SKIP).to(device)
target_net = DQN(66,66,n_actions,ACTION_SKIP).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

# init optimizer

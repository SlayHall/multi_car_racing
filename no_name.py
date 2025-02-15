هذا كود ال main
def main():
    global epsilon
    num_episodes = 4000
    env = gym.make(
        "MultiCarRacing-v0",
        num_agents=num_agents,
        direction='CCW',
        use_random_direction=True,
        backwards_flag=True,
        h_ratio=0.25,
        use_ego_color=False
    )
    preprocessors = [Preprocessor(frame_stack_size=4, frame_skip=4) for _ in range(num_agents)]
    for episode in range(num_episodes):
        obs = env.reset()
        for agent_idx, preproc in enumerate(preprocessors):
            preproc.reset()
            preproc.preprocess(obs[agent_idx])
        done = False
        total_reward = [0.0] * num_agents
        timestep = 0
        while not done and timestep < max_timesteps:
            current_states = [preprocessors[i].get_state() for i in range(num_agents)]
            if episode >= 1000:
                epsilon = 0
                env.render()
            discrete_actions = []
            env_actions = []
            for agent_idx in range(num_agents):
                state_tensor = torch.tensor(current_states[agent_idx], dtype=torch.float32).to(device)
                discrete_action, continuous_action = select_action(agent_idx, state_tensor, epsilon)
                discrete_actions.append(discrete_action)
                env_actions.append(continuous_action)
            env_actions_np = np.array(env_actions)
            skip_reward = [0.0] * num_agents
            for skip in range(preprocessors[0].frame_skip):
                next_obs, reward, done, info = env.step(env_actions_np)
                skip_reward = [prev + r for prev, r in zip(skip_reward, reward)]
                for agent_idx in range(num_agents):
                    preprocessors[agent_idx].preprocess(next_obs[agent_idx])
                if done:
                    break
            next_states = [preprocessors[i].get_state() for i in range(num_agents)]
            for agent_idx in range(num_agents):
                replay_buffers[agent_idx].add(
                    current_states[agent_idx],
                    discrete_actions[agent_idx],
                    skip_reward[agent_idx],
                    next_states[agent_idx],
                    done
                )
                total_reward[agent_idx] += skip_reward[agent_idx]
            obs = next_obs
            timestep += 1
            for agent_idx in range(num_agents):
                if len(replay_buffers[agent_idx]) >= batch_size:
                    loss = train_step(agent_idx, replay_buffers[agent_idx])
        if (episode + 1) % update_target_every == 0:
            for agent_idx in range(num_agents):
                update_target_network(agent_idx)
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        print(f"Episode {episode + 1}/{num_episodes}, Rewards: {total_reward}, Epsilon: {epsilon:.5f}")
    env.close()

if name == "main":
    main()
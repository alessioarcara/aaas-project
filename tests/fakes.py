def fake_policy(obs):
    batch_size = obs["agent_0_obs"].shape[0]
    actions = [(0, 0)] * batch_size  # stay
    return actions

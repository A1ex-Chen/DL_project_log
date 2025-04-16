def __call__(self, obs, batch_size=64, planning_horizon=32, n_guide_steps=2,
    scale=0.1):
    obs = self.normalize(obs, 'observations')
    obs = obs[None].repeat(batch_size, axis=0)
    conditions = {(0): self.to_torch(obs)}
    shape = batch_size, planning_horizon, self.state_dim + self.action_dim
    x1 = randn_tensor(shape, device=self.unet.device)
    x = self.reset_x0(x1, conditions, self.action_dim)
    x = self.to_torch(x)
    x, y = self.run_diffusion(x, conditions, n_guide_steps, scale)
    sorted_idx = y.argsort(0, descending=True).squeeze()
    sorted_values = x[sorted_idx]
    actions = sorted_values[:, :, :self.action_dim]
    actions = actions.detach().cpu().numpy()
    denorm_actions = self.de_normalize(actions, key='actions')
    if y is not None:
        selected_index = 0
    else:
        selected_index = np.random.randint(0, batch_size)
    denorm_actions = denorm_actions[selected_index, 0]
    return denorm_actions

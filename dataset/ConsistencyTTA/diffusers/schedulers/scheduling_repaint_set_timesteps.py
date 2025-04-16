def set_timesteps(self, num_inference_steps: int, jump_length: int=10,
    jump_n_sample: int=10, device: Union[str, torch.device]=None):
    num_inference_steps = min(self.config.num_train_timesteps,
        num_inference_steps)
    self.num_inference_steps = num_inference_steps
    timesteps = []
    jumps = {}
    for j in range(0, num_inference_steps - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1
    t = num_inference_steps
    while t >= 1:
        t = t - 1
        timesteps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                timesteps.append(t)
    timesteps = np.array(timesteps) * (self.config.num_train_timesteps //
        self.num_inference_steps)
    self.timesteps = torch.from_numpy(timesteps).to(device)

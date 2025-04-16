def previous_timestep(self, timestep):
    if self.custom_timesteps:
        index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
        if index == self.timesteps.shape[0] - 1:
            prev_t = torch.tensor(-1)
        else:
            prev_t = self.timesteps[index + 1]
    else:
        num_inference_steps = (self.num_inference_steps if self.
            num_inference_steps else self.config.num_train_timesteps)
        prev_t = (timestep - self.config.num_train_timesteps //
            num_inference_steps)
    return prev_t

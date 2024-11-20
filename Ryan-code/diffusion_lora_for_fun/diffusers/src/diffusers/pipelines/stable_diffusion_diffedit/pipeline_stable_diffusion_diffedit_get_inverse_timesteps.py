def get_inverse_timesteps(self, num_inference_steps, strength, device):
    init_timestep = min(int(num_inference_steps * strength),
        num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    if t_start == 0:
        return self.inverse_scheduler.timesteps, num_inference_steps
    timesteps = self.inverse_scheduler.timesteps[:-t_start]
    return timesteps, num_inference_steps - t_start

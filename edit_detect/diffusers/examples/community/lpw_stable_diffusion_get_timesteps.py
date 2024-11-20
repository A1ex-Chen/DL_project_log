def get_timesteps(self, num_inference_steps, strength, device, is_text2img):
    if is_text2img:
        return self.scheduler.timesteps.to(device), num_inference_steps
    else:
        init_timestep = min(int(num_inference_steps * strength),
            num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        return timesteps, num_inference_steps - t_start
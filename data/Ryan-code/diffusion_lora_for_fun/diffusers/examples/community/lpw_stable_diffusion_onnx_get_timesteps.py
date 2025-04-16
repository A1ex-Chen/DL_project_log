def get_timesteps(self, num_inference_steps, strength, is_text2img):
    if is_text2img:
        return self.scheduler.timesteps, num_inference_steps
    else:
        offset = self.scheduler.config.get('steps_offset', 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        return timesteps, num_inference_steps - t_start

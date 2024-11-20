def get_latest_timestep_img2img(self, num_inference_steps, strength):
    """Finds the latest timesteps where an img2img strength does not impose latents anymore"""
    offset = self.scheduler.config.get('steps_offset', 0)
    init_timestep = int(num_inference_steps * (1 - strength)) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    t_start = min(max(num_inference_steps - init_timestep + offset, 0), 
        num_inference_steps - 1)
    latest_timestep = self.scheduler.timesteps[t_start]
    return latest_timestep

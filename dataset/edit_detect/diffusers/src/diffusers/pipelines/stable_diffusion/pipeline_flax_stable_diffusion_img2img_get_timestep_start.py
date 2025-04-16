def get_timestep_start(self, num_inference_steps, strength):
    init_timestep = min(int(num_inference_steps * strength),
        num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    return t_start

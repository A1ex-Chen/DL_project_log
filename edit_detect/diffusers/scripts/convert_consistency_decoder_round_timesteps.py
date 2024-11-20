@staticmethod
def round_timesteps(timesteps, total_timesteps, n_distilled_steps,
    truncate_start=True):
    with torch.no_grad():
        space = torch.div(total_timesteps, n_distilled_steps, rounding_mode
            ='floor')
        rounded_timesteps = (torch.div(timesteps, space, rounding_mode=
            'floor') + 1) * space
        if truncate_start:
            rounded_timesteps[rounded_timesteps == total_timesteps] -= space
        else:
            rounded_timesteps[rounded_timesteps == total_timesteps] -= space
            rounded_timesteps[rounded_timesteps == 0] += space
        return rounded_timesteps

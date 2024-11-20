@register_to_config
def __init__(self, num_train_timesteps: int=1000, variance_type: str=
    'fixed_small_log', clip_sample: bool=True, clip_sample_range: Optional[
    float]=1.0, prediction_type: str='epsilon', beta_schedule: str=
    'squaredcos_cap_v2'):
    if beta_schedule != 'squaredcos_cap_v2':
        raise ValueError(
            "UnCLIPScheduler only supports `beta_schedule`: 'squaredcos_cap_v2'"
            )
    self.betas = betas_for_alpha_bar(num_train_timesteps)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.one = torch.tensor(1.0)
    self.init_noise_sigma = 1.0
    self.num_inference_steps = None
    self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-
        1].copy())
    self.variance_type = variance_type

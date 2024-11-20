@register_to_config
def __init__(self, num_train_timesteps=2000, beta_min=0.1, beta_max=20,
    sampling_eps=0.001):
    self.sigmas = None
    self.discrete_sigmas = None
    self.timesteps = None

@register_to_config
def __init__(self, sigma_min: float=0.02, sigma_max: float=100, s_noise:
    float=1.007, s_churn: float=80, s_min: float=0.05, s_max: float=50):
    self.init_noise_sigma = sigma_max
    self.num_inference_steps: int = None
    self.timesteps: np.IntTensor = None
    self.schedule: torch.FloatTensor = None

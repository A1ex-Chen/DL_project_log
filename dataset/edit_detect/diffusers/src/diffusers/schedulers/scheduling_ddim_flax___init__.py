@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[jnp.ndarray]=None, clip_sample: bool=True, clip_sample_range:
    float=1.0, set_alpha_to_one: bool=True, steps_offset: int=0,
    prediction_type: str='epsilon', dtype: jnp.dtype=jnp.float32):
    self.dtype = dtype

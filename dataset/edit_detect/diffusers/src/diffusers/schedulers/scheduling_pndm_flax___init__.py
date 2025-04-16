@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[jnp.ndarray]=None, skip_prk_steps: bool=False,
    set_alpha_to_one: bool=False, steps_offset: int=0, prediction_type: str
    ='epsilon', dtype: jnp.dtype=jnp.float32):
    self.dtype = dtype
    self.pndm_order = 4

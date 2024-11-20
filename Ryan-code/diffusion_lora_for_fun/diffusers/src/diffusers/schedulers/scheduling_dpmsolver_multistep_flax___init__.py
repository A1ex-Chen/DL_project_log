@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[jnp.ndarray]=None, solver_order: int=2, prediction_type: str=
    'epsilon', thresholding: bool=False, dynamic_thresholding_ratio: float=
    0.995, sample_max_value: float=1.0, algorithm_type: str='dpmsolver++',
    solver_type: str='midpoint', lower_order_final: bool=True,
    timestep_spacing: str='linspace', dtype: jnp.dtype=jnp.float32):
    self.dtype = dtype

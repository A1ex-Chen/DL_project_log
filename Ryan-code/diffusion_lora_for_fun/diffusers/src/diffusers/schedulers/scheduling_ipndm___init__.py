@register_to_config
def __init__(self, num_train_timesteps: int=1000, trained_betas: Optional[
    Union[np.ndarray, List[float]]]=None):
    self.set_timesteps(num_train_timesteps)
    self.init_noise_sigma = 1.0
    self.pndm_order = 4
    self.ets = []
    self._step_index = None
    self._begin_index = None

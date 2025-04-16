def __init__(self, model, schedule='linear', **kwargs):
    super().__init__()
    self.model = model
    self.ddpm_num_timesteps = model.num_timesteps
    self.schedule = schedule

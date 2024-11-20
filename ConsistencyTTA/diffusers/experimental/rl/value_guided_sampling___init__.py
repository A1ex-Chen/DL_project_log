def __init__(self, value_function: UNet1DModel, unet: UNet1DModel,
    scheduler: DDPMScheduler, env):
    super().__init__()
    self.value_function = value_function
    self.unet = unet
    self.scheduler = scheduler
    self.env = env
    self.data = env.get_dataset()
    self.means = {}
    for key in self.data.keys():
        try:
            self.means[key] = self.data[key].mean()
        except:
            pass
    self.stds = {}
    for key in self.data.keys():
        try:
            self.stds[key] = self.data[key].std()
        except:
            pass
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]

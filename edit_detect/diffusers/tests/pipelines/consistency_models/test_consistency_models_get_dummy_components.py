def get_dummy_components(self, class_cond=False):
    if class_cond:
        unet = self.dummy_cond_unet
    else:
        unet = self.dummy_uncond_unet
    scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40,
        sigma_min=0.002, sigma_max=80.0)
    components = {'unet': unet, 'scheduler': scheduler}
    return components

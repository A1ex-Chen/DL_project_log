def get_dummy_components(self):
    unet = self.dummy_unet
    movq = self.dummy_movq
    ddim_config = {'num_train_timesteps': 1000, 'beta_schedule': 'linear',
        'beta_start': 0.00085, 'beta_end': 0.012, 'clip_sample': False,
        'set_alpha_to_one': False, 'steps_offset': 0, 'prediction_type':
        'epsilon', 'thresholding': False}
    scheduler = DDIMScheduler(**ddim_config)
    components = {'unet': unet, 'scheduler': scheduler, 'movq': movq}
    return components

def get_dummy_components(self):
    prior = self.dummy_prior
    image_encoder = self.dummy_image_encoder
    image_processor = self.dummy_image_processor
    shap_e_renderer = self.dummy_renderer
    scheduler = HeunDiscreteScheduler(beta_schedule='exp',
        num_train_timesteps=1024, prediction_type='sample',
        use_karras_sigmas=True, clip_sample=True, clip_sample_range=1.0)
    components = {'prior': prior, 'image_encoder': image_encoder,
        'image_processor': image_processor, 'shap_e_renderer':
        shap_e_renderer, 'scheduler': scheduler}
    return components

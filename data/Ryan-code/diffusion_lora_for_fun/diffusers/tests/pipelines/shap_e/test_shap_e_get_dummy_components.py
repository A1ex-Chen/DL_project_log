def get_dummy_components(self):
    prior = self.dummy_prior
    text_encoder = self.dummy_text_encoder
    tokenizer = self.dummy_tokenizer
    shap_e_renderer = self.dummy_renderer
    scheduler = HeunDiscreteScheduler(beta_schedule='exp',
        num_train_timesteps=1024, prediction_type='sample',
        use_karras_sigmas=True, clip_sample=True, clip_sample_range=1.0)
    components = {'prior': prior, 'text_encoder': text_encoder, 'tokenizer':
        tokenizer, 'shap_e_renderer': shap_e_renderer, 'scheduler': scheduler}
    return components

def get_dummy_components(self):
    prior = self.dummy_prior
    image_encoder = self.dummy_image_encoder
    text_encoder = self.dummy_text_encoder
    tokenizer = self.dummy_tokenizer
    image_processor = self.dummy_image_processor
    scheduler = UnCLIPScheduler(variance_type='fixed_small_log',
        prediction_type='sample', num_train_timesteps=1000, clip_sample=
        True, clip_sample_range=10.0)
    components = {'prior': prior, 'image_encoder': image_encoder,
        'text_encoder': text_encoder, 'tokenizer': tokenizer, 'scheduler':
        scheduler, 'image_processor': image_processor}
    return components

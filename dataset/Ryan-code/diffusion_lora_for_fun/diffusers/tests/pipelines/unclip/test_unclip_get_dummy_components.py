def get_dummy_components(self):
    prior = self.dummy_prior
    decoder = self.dummy_decoder
    text_proj = self.dummy_text_proj
    text_encoder = self.dummy_text_encoder
    tokenizer = self.dummy_tokenizer
    super_res_first = self.dummy_super_res_first
    super_res_last = self.dummy_super_res_last
    prior_scheduler = UnCLIPScheduler(variance_type='fixed_small_log',
        prediction_type='sample', num_train_timesteps=1000,
        clip_sample_range=5.0)
    decoder_scheduler = UnCLIPScheduler(variance_type='learned_range',
        prediction_type='epsilon', num_train_timesteps=1000)
    super_res_scheduler = UnCLIPScheduler(variance_type='fixed_small_log',
        prediction_type='epsilon', num_train_timesteps=1000)
    components = {'prior': prior, 'decoder': decoder, 'text_proj':
        text_proj, 'text_encoder': text_encoder, 'tokenizer': tokenizer,
        'super_res_first': super_res_first, 'super_res_last':
        super_res_last, 'prior_scheduler': prior_scheduler,
        'decoder_scheduler': decoder_scheduler, 'super_res_scheduler':
        super_res_scheduler}
    return components

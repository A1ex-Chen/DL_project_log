def get_dummy_components(self):
    decoder = self.dummy_decoder
    text_proj = self.dummy_text_proj
    text_encoder = self.dummy_text_encoder
    tokenizer = self.dummy_tokenizer
    super_res_first = self.dummy_super_res_first
    super_res_last = self.dummy_super_res_last
    decoder_scheduler = UnCLIPScheduler(variance_type='learned_range',
        prediction_type='epsilon', num_train_timesteps=1000)
    super_res_scheduler = UnCLIPScheduler(variance_type='fixed_small_log',
        prediction_type='epsilon', num_train_timesteps=1000)
    feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
    image_encoder = self.dummy_image_encoder
    return {'decoder': decoder, 'text_encoder': text_encoder, 'tokenizer':
        tokenizer, 'text_proj': text_proj, 'feature_extractor':
        feature_extractor, 'image_encoder': image_encoder,
        'super_res_first': super_res_first, 'super_res_last':
        super_res_last, 'decoder_scheduler': decoder_scheduler,
        'super_res_scheduler': super_res_scheduler}

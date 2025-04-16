def get_dummy_components(self):
    prior = self.dummy_prior
    scheduler = DDPMWuerstchenScheduler()
    tokenizer = self.dummy_tokenizer
    text_encoder = self.dummy_text_encoder
    decoder = self.dummy_decoder
    vqgan = self.dummy_vqgan
    prior_text_encoder = self.dummy_text_encoder
    prior_tokenizer = self.dummy_tokenizer
    components = {'text_encoder': text_encoder, 'tokenizer': tokenizer,
        'decoder': decoder, 'scheduler': scheduler, 'vqgan': vqgan,
        'prior_text_encoder': prior_text_encoder, 'prior_tokenizer':
        prior_tokenizer, 'prior_prior': prior, 'prior_scheduler': scheduler,
        'prior_feature_extractor': None, 'prior_image_encoder': None}
    return components

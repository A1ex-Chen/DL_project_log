def get_dummy_components(self):
    prior = self.dummy_prior
    prior_text_encoder = self.dummy_prior_text_encoder
    scheduler = DDPMWuerstchenScheduler()
    tokenizer = self.dummy_tokenizer
    text_encoder = self.dummy_text_encoder
    decoder = self.dummy_decoder
    vqgan = self.dummy_vqgan
    components = {'tokenizer': tokenizer, 'text_encoder': text_encoder,
        'decoder': decoder, 'vqgan': vqgan, 'scheduler': scheduler,
        'prior_prior': prior, 'prior_text_encoder': prior_text_encoder,
        'prior_tokenizer': tokenizer, 'prior_scheduler': scheduler}
    return components

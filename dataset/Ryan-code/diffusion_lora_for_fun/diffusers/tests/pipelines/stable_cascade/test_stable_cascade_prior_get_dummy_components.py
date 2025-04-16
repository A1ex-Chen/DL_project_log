def get_dummy_components(self):
    prior = self.dummy_prior
    text_encoder = self.dummy_text_encoder
    tokenizer = self.dummy_tokenizer
    scheduler = DDPMWuerstchenScheduler()
    components = {'prior': prior, 'text_encoder': text_encoder, 'tokenizer':
        tokenizer, 'scheduler': scheduler, 'feature_extractor': None,
        'image_encoder': None}
    return components

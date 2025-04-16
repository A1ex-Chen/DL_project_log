def get_dummy_components(self):
    decoder = self.dummy_decoder
    text_encoder = self.dummy_text_encoder
    tokenizer = self.dummy_tokenizer
    vqgan = self.dummy_vqgan
    scheduler = DDPMWuerstchenScheduler()
    components = {'decoder': decoder, 'vqgan': vqgan, 'text_encoder':
        text_encoder, 'tokenizer': tokenizer, 'scheduler': scheduler,
        'latent_dim_scale': 4.0}
    return components

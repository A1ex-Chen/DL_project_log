def test_optional_components_is_none(self):
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    items = {'feature_extractor': self.dummy_extractor, 'unet': unet,
        'scheduler': scheduler, 'vae': vae, 'text_encoder': bert,
        'tokenizer': tokenizer, 'safety_checker': None}
    pipeline = StableDiffusionPipeline(**items)
    assert sorted(pipeline.components.keys()) == sorted(['image_encoder'] +
        list(items.keys()))
    assert pipeline.image_encoder is None

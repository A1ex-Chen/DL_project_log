def test_set_component_to_none(self):
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    pipeline = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=
        vae, text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'This is a flower'
    out_image = pipeline(prompt=prompt, generator=generator,
        num_inference_steps=1, output_type='np').images
    pipeline.feature_extractor = None
    generator = torch.Generator(device='cpu').manual_seed(0)
    out_image_2 = pipeline(prompt=prompt, generator=generator,
        num_inference_steps=1, output_type='np').images
    assert out_image.shape == (1, 64, 64, 3)
    assert np.abs(out_image - out_image_2).max() < 0.001

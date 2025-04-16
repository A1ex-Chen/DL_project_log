@unittest.skipIf(torch_device != 'cuda', 'This test requires a GPU')
def test_stable_diffusion_fp16(self):
    """Test that stable diffusion works with fp16"""
    unet = self.dummy_cond_unet
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    unet = unet.half()
    vae = vae.half()
    bert = bert.half()
    sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=
        vae, text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    image = sd_pipe([prompt], num_inference_steps=2, output_type='np').images
    assert image.shape == (1, 64, 64, 3)

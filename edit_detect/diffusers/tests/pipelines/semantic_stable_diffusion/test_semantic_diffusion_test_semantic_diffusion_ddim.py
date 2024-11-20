def test_semantic_diffusion_ddim(self):
    device = 'cpu'
    unet = self.dummy_cond_unet
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=
        vae, text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.Generator(device=device).manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=6.0,
        num_inference_steps=2, output_type='np')
    image = output.images
    generator = torch.Generator(device=device).manual_seed(0)
    image_from_tuple = sd_pipe([prompt], generator=generator,
        guidance_scale=6.0, num_inference_steps=2, output_type='np',
        return_dict=False)[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5753, 0.6114, 0.5001, 0.5034, 0.547, 
        0.4729, 0.4971, 0.4867, 0.4867])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01

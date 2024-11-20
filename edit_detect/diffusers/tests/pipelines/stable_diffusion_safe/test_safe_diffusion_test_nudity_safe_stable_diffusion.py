def test_nudity_safe_stable_diffusion(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None)
    sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.
        config)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'padme amidala taking a bath artwork, safe for work, no nudity'
    seed = 2734971755
    guidance_scale = 7
    generator = torch.manual_seed(seed)
    output = sd_pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, sld_guidance_scale=0)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.3502, 0.3622, 0.3396, 0.3642, 0.3478, 0.3318, 0.35,
        0.3348, 0.3297]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    generator = torch.manual_seed(seed)
    output = sd_pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, sld_guidance_scale=2000, sld_warmup_steps=7,
        sld_threshold=0.025, sld_momentum_scale=0.5, sld_mom_beta=0.7)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.5531, 0.5206, 0.4895, 0.5156, 0.5182, 0.4751, 
        0.4802, 0.4803, 0.4443]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

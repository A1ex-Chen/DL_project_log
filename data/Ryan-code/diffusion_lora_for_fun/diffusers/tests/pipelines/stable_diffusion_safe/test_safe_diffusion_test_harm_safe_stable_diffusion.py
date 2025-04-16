def test_harm_safe_stable_diffusion(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None)
    sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.
        config)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = (
        'portrait of girl with smokey eyes makeup in abandoned hotel, grange clothes, redshift, wide high angle coloured polaroid photograph with flash, kodak film, hyper real, stunning moody cinematography, with anamorphic lenses, by maripol, fallen angels by wong kar - wai, style of suspiria and neon demon and children from bahnhof zoo, detailed '
        )
    seed = 4003660346
    guidance_scale = 7
    generator = torch.manual_seed(seed)
    output = sd_pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, sld_guidance_scale=0)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.2278, 0.2231, 0.2249, 0.2333, 0.2303, 0.1885, 
        0.2273, 0.2144, 0.2176]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    generator = torch.manual_seed(seed)
    output = sd_pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, sld_guidance_scale=2000, sld_warmup_steps=7,
        sld_threshold=0.025, sld_momentum_scale=0.5, sld_mom_beta=0.7)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.2383, 0.2276, 0.236, 0.2192, 0.2186, 0.2053, 0.1971,
        0.1901, 0.1719]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

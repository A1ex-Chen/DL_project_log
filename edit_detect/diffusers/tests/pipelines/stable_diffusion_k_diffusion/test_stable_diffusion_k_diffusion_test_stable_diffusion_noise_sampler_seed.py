def test_stable_diffusion_noise_sampler_seed(self):
    sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4')
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    sd_pipe.set_scheduler('sample_dpmpp_sde')
    prompt = 'A painting of a squirrel eating a burger'
    seed = 0
    images1 = sd_pipe([prompt], generator=torch.manual_seed(seed),
        noise_sampler_seed=seed, guidance_scale=9.0, num_inference_steps=20,
        output_type='np').images
    images2 = sd_pipe([prompt], generator=torch.manual_seed(seed),
        noise_sampler_seed=seed, guidance_scale=9.0, num_inference_steps=20,
        output_type='np').images
    assert images1.shape == (1, 512, 512, 3)
    assert images2.shape == (1, 512, 512, 3)
    assert np.abs(images1.flatten() - images2.flatten()).max() < 0.01

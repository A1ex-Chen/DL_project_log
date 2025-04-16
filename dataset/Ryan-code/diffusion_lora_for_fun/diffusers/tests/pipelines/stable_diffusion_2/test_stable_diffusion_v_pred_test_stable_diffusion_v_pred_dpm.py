def test_stable_diffusion_v_pred_dpm(self):
    """
        TODO: update this test after making DPM compatible with V-prediction!
        """
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        'stabilityai/stable-diffusion-2', subfolder='scheduler',
        final_sigmas_type='sigma_min')
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2', scheduler=scheduler)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.enable_attention_slicing()
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'a photograph of an astronaut riding a horse'
    generator = torch.manual_seed(0)
    image = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
        num_inference_steps=5, output_type='np').images
    image_slice = image[0, 253:256, 253:256, -1]
    assert image.shape == (1, 768, 768, 3)
    expected_slice = np.array([0.3303, 0.3184, 0.3291, 0.33, 0.3256, 0.3113,
        0.2965, 0.3134, 0.3192])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

def test_stable_diffusion_2(self):
    sag_pipe = StableDiffusionSAGPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base')
    sag_pipe = sag_pipe.to(torch_device)
    sag_pipe.set_progress_bar_config(disable=None)
    prompt = '.'
    generator = torch.manual_seed(0)
    output = sag_pipe([prompt], generator=generator, guidance_scale=7.5,
        sag_scale=1.0, num_inference_steps=20, output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.3459, 0.2876, 0.2537, 0.3002, 0.2671, 
        0.216, 0.3026, 0.2262, 0.2371])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

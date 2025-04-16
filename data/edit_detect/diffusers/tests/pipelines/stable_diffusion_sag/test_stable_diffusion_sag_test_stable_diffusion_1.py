def test_stable_diffusion_1(self):
    sag_pipe = StableDiffusionSAGPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4')
    sag_pipe = sag_pipe.to(torch_device)
    sag_pipe.set_progress_bar_config(disable=None)
    prompt = '.'
    generator = torch.manual_seed(0)
    output = sag_pipe([prompt], generator=generator, guidance_scale=7.5,
        sag_scale=1.0, num_inference_steps=20, output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.1568, 0.1738, 0.1695, 0.1693, 0.1507, 
        0.1705, 0.1547, 0.1751, 0.1949])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

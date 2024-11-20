def test_stable_diffusion_1(self):
    sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4')
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    sd_pipe.set_scheduler('sample_euler')
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=9.0,
        num_inference_steps=20, output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.0447, 0.0492, 0.0468, 0.0408, 0.0383, 
        0.0408, 0.0354, 0.038, 0.0339])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

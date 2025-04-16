def test_stable_diffusion_xl(self):
    sd_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=self.dtype)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    sd_pipe.set_scheduler('sample_euler')
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=9.0,
        num_inference_steps=20, height=512, width=512, output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.79600024, 0.796546, 0.80682373, 0.79428387,
        0.7905743, 0.8008807, 0.786183, 0.7835959, 0.797892])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

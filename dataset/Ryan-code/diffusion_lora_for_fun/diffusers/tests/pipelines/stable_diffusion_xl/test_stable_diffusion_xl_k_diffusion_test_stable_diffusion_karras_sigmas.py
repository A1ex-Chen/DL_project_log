def test_stable_diffusion_karras_sigmas(self):
    sd_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=self.dtype)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    sd_pipe.set_scheduler('sample_dpmpp_2m')
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
        num_inference_steps=15, output_type='np', use_karras_sigmas=True,
        height=512, width=512)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.9506951, 0.9527786, 0.95309967, 0.9511477,
        0.952523, 0.9515326, 0.9511933, 0.9480397, 0.94930184])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

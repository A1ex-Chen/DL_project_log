def test_stable_diffusion_2(self):
    sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base')
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
    expected_slice = np.array([0.1237, 0.132, 0.1438, 0.1359, 0.139, 0.1132,
        0.1277, 0.1175, 0.1112])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.5

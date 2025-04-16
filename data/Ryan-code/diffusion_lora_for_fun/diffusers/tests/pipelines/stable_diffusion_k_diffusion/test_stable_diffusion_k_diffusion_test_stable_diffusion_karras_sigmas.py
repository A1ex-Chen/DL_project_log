def test_stable_diffusion_karras_sigmas(self):
    sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base')
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    sd_pipe.set_scheduler('sample_dpmpp_2m')
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
        num_inference_steps=15, output_type='np', use_karras_sigmas=True)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.11381689, 0.12112921, 0.1389457, 
        0.12549606, 0.1244964, 0.10831517, 0.11562866, 0.10867816, 0.10499048])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

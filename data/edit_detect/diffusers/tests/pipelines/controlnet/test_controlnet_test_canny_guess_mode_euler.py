def test_canny_guess_mode_euler(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, controlnet=
        controlnet)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = ''
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        )
    output = pipe(prompt, image, generator=generator, output_type='np',
        num_inference_steps=3, guidance_scale=3.0, guess_mode=True)
    image = output.images[0]
    assert image.shape == (768, 512, 3)
    image_slice = image[-3:, -3:, -1]
    expected_slice = np.array([0.1655, 0.1721, 0.1623, 0.1685, 0.1711, 
        0.1646, 0.1651, 0.1631, 0.1494])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

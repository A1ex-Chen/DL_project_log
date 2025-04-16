def test_v11_shuffle_global_pool_conditions(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11e_sd15_shuffle')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, controlnet=
        controlnet)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'New York'
    image = load_image(
        'https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/control.png'
        )
    output = pipe(prompt, image, generator=generator, output_type='np',
        num_inference_steps=3, guidance_scale=7.0)
    image = output.images[0]
    assert image.shape == (512, 640, 3)
    image_slice = image[-3:, -3:, -1]
    expected_slice = np.array([0.1338, 0.1597, 0.1202, 0.1687, 0.1377, 
        0.1017, 0.207, 0.1574, 0.1348])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

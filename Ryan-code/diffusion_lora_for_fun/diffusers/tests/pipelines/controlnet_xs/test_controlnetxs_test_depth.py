def test_depth(self):
    controlnet = ControlNetXSAdapter.from_pretrained(
        'UmerHA/Testing-ConrolNetXS-SD2.1-depth', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base', controlnet=controlnet,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = "Stormtrooper's lecture"
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png'
        )
    output = pipe(prompt, image, generator=generator, output_type='np',
        num_inference_steps=3)
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    original_image = image[-3:, -3:, -1].flatten()
    expected_image = np.array([0.4844, 0.4937, 0.4956, 0.4663, 0.5039, 
        0.5044, 0.4565, 0.4883, 0.4941])
    assert np.allclose(original_image, expected_image, atol=0.0001)

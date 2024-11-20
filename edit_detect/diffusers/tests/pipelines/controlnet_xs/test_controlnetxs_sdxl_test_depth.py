def test_depth(self):
    controlnet = ControlNetXSAdapter.from_pretrained(
        'UmerHA/Testing-ConrolNetXS-SDXL-depth', torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet,
        torch_dtype=torch.float16)
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = "Stormtrooper's lecture"
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png'
        )
    images = pipe(prompt, image=image, generator=generator, output_type=
        'np', num_inference_steps=3).images
    assert images[0].shape == (512, 512, 3)
    original_image = images[0, -3:, -3:, -1].flatten()
    expected_image = np.array([0.5448, 0.5437, 0.5426, 0.5543, 0.553, 
        0.5475, 0.5595, 0.5602, 0.5529])
    assert np.allclose(original_image, expected_image, atol=0.0001)

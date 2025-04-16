def test_canny(self):
    controlnet = ControlNetXSAdapter.from_pretrained(
        'UmerHA/Testing-ConrolNetXS-SD2.1-canny', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base', controlnet=controlnet,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'bird'
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        )
    output = pipe(prompt, image, generator=generator, output_type='np',
        num_inference_steps=3)
    image = output.images[0]
    assert image.shape == (768, 512, 3)
    original_image = image[-3:, -3:, -1].flatten()
    expected_image = np.array([0.1963, 0.229, 0.2659, 0.2109, 0.2332, 
        0.2827, 0.2534, 0.2422, 0.2808])
    assert np.allclose(original_image, expected_image, atol=0.0001)

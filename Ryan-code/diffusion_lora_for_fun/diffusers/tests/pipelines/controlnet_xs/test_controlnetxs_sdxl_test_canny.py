def test_canny(self):
    controlnet = ControlNetXSAdapter.from_pretrained(
        'UmerHA/Testing-ConrolNetXS-SDXL-canny', torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet,
        torch_dtype=torch.float16)
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'bird'
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        )
    images = pipe(prompt, image=image, generator=generator, output_type=
        'np', num_inference_steps=3).images
    assert images[0].shape == (768, 512, 3)
    original_image = images[0, -3:, -3:, -1].flatten()
    expected_image = np.array([0.3202, 0.3151, 0.3328, 0.3172, 0.337, 
        0.3381, 0.3378, 0.3389, 0.3224])
    assert np.allclose(original_image, expected_image, atol=0.0001)

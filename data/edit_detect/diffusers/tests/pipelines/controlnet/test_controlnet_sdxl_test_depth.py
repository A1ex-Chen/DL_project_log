def test_depth(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-depth-sdxl-1.0')
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet)
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
    expected_image = np.array([0.4399, 0.5112, 0.5478, 0.4314, 0.472, 
        0.4823, 0.4647, 0.4957, 0.4853])
    assert np.allclose(original_image, expected_image, atol=0.0001)

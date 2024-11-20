def test_canny(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-canny-sdxl-1.0')
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet)
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
    expected_image = np.array([0.4185, 0.4127, 0.4089, 0.4046, 0.4115, 
        0.4096, 0.4081, 0.4112, 0.3913])
    assert np.allclose(original_image, expected_image, atol=0.0001)

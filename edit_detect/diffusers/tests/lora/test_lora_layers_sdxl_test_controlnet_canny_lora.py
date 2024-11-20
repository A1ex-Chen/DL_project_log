def test_controlnet_canny_lora(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-canny-sdxl-1.0')
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet)
    pipe.load_lora_weights('nerijs/pixel-art-xl', weight_name=
        'pixel-art-xl.safetensors')
    pipe.enable_sequential_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'corgi'
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        )
    images = pipe(prompt, image=image, generator=generator, output_type=
        'np', num_inference_steps=3).images
    assert images[0].shape == (768, 512, 3)
    original_image = images[0, -3:, -3:, -1].flatten()
    expected_image = np.array([0.4574, 0.4461, 0.4435, 0.4462, 0.4396, 
        0.439, 0.4474, 0.4486, 0.4333])
    max_diff = numpy_cosine_similarity_distance(expected_image, original_image)
    assert max_diff < 0.0001
    pipe.unload_lora_weights()
    release_memory(pipe)

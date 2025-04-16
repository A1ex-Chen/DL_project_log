def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
    latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
    init_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png'
        )
    init_image = init_image.resize((512, 512))
    inputs = {'prompt': 'a photograph of an astronaut riding a horse',
        'latents': latents, 'generator': generator, 'num_inference_steps': 
        3, 'guidance_scale': 7.5, 'output_type': 'np', 'image': init_image}
    return inputs

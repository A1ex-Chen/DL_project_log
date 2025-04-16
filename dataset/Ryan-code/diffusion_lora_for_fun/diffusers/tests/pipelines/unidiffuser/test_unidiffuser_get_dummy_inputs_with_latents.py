def get_dummy_inputs_with_latents(self, device, seed=0):
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg'
        )
    image = image.resize((32, 32))
    latents = self.get_fixed_latents(device, seed=seed)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'an elephant under the sea', 'image': image,
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np', 'prompt_latents': latents.get(
        'prompt_latents'), 'vae_latents': latents.get('vae_latents'),
        'clip_latents': latents.get('clip_latents')}
    return inputs

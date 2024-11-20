def get_inputs(self, device, seed=0, generate_latents=False):
    generator = torch.manual_seed(seed)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg'
        )
    inputs = {'prompt': 'an elephant under the sea', 'image': image,
        'generator': generator, 'num_inference_steps': 3, 'guidance_scale':
        8.0, 'output_type': 'np'}
    if generate_latents:
        latents = self.get_fixed_latents(device, seed=seed)
        for latent_name, latent_tensor in latents.items():
            inputs[latent_name] = latent_tensor
    return inputs

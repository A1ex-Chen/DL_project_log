def get_inputs(self, seed=0):
    generator = torch.manual_seed(seed)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_pix2pix/example.jpg'
        )
    inputs = {'prompt': 'turn him into a cyborg', 'image': image,
        'generator': generator, 'num_inference_steps': 3, 'guidance_scale':
        7.5, 'image_guidance_scale': 1.0, 'output_type': 'np'}
    return inputs

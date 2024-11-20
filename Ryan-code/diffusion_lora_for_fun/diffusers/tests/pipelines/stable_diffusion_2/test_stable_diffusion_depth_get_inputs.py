def get_inputs(self, device='cpu', dtype=torch.float32, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png'
        )
    inputs = {'prompt': 'two tigers', 'image': init_image, 'generator':
        generator, 'num_inference_steps': 3, 'strength': 0.75,
        'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

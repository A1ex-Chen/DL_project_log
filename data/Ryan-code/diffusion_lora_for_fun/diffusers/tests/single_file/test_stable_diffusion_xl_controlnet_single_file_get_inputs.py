def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png'
        )
    inputs = {'prompt': "Stormtrooper's lecture", 'image': image,
        'generator': generator, 'num_inference_steps': 2, 'strength': 0.75,
        'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

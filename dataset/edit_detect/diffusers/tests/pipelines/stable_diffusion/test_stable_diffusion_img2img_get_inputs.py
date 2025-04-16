def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    init_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png'
        )
    inputs = {'prompt': 'a fantasy landscape, concept art, high resolution',
        'image': init_image, 'generator': generator, 'num_inference_steps':
        50, 'strength': 0.75, 'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

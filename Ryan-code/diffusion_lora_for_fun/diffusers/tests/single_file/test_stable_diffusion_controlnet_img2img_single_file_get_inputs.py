def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    init_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png'
        )
    control_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        ).resize((512, 512))
    prompt = 'bird'
    inputs = {'prompt': prompt, 'image': init_image, 'control_image':
        control_image, 'generator': generator, 'num_inference_steps': 3,
        'strength': 0.75, 'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    init_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_image.png'
        )
    mask_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_mask.png'
        )
    inputs = {'prompt':
        'Face of a yellow cat, high resolution, sitting on a park bench',
        'image': init_image, 'mask_image': mask_image, 'generator':
        generator, 'num_inference_steps': 50, 'guidance_scale': 7.5,
        'output_type': 'np'}
    return inputs

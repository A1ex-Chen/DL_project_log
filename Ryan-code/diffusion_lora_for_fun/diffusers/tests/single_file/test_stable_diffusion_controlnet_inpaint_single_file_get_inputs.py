def get_inputs(self):
    control_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        ).resize((512, 512))
    image = load_image(
        'https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png'
        ).resize((512, 512))
    mask_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_mask.png'
        ).resize((512, 512))
    inputs = {'prompt': 'bird', 'image': image, 'control_image':
        control_image, 'mask_image': mask_image, 'generator': torch.
        Generator(device='cpu').manual_seed(0), 'num_inference_steps': 3,
        'output_type': 'np'}
    return inputs

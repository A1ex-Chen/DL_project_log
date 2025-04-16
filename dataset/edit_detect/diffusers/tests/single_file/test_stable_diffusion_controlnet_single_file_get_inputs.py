def get_inputs(self):
    control_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        ).resize((512, 512))
    inputs = {'prompt': 'bird', 'image': control_image, 'generator': torch.
        Generator(device='cpu').manual_seed(0), 'num_inference_steps': 3,
        'output_type': 'np'}
    return inputs

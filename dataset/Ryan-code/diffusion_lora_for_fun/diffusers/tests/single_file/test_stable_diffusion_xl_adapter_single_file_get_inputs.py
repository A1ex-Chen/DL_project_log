def get_inputs(self):
    prompt = 'toy'
    generator = torch.Generator(device='cpu').manual_seed(0)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png'
        )
    inputs = {'prompt': prompt, 'image': image, 'generator': generator,
        'num_inference_steps': 2, 'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

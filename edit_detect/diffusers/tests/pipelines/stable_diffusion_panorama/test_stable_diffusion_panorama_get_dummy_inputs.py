def get_dummy_inputs(self, device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'a photo of the dolomites', 'generator': generator,
        'height': None, 'width': None, 'num_inference_steps': 1,
        'guidance_scale': 6.0, 'output_type': 'np'}
    return inputs

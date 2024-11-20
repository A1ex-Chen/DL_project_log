def get_inputs(self, seed=0):
    generator = torch.manual_seed(seed)
    inputs = {'prompt': 'a photo of the dolomites', 'generator': generator,
        'num_inference_steps': 3, 'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

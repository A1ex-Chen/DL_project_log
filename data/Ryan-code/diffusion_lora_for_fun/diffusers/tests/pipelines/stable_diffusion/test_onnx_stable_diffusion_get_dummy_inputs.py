def get_dummy_inputs(self, seed=0):
    generator = np.random.RandomState(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        7.5, 'output_type': 'np'}
    return inputs

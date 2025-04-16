def get_dummy_inputs(self, seed=0):
    image = floats_tensor((1, 3, 128, 128), rng=random.Random(seed))
    generator = np.random.RandomState(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger', 'image':
        image, 'generator': generator, 'num_inference_steps': 3,
        'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs

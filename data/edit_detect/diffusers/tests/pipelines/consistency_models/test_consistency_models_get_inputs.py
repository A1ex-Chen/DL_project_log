def get_inputs(self, seed=0, get_fixed_latents=False, device='cpu', dtype=
    torch.float32, shape=(1, 3, 64, 64)):
    generator = torch.manual_seed(seed)
    inputs = {'num_inference_steps': None, 'timesteps': [22, 0],
        'class_labels': 0, 'generator': generator, 'output_type': 'np'}
    if get_fixed_latents:
        latents = self.get_fixed_latents(seed=seed, device=device, dtype=
            dtype, shape=shape)
        inputs['latents'] = latents
    return inputs

def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'generator': generator, 'editing_prompt': ['wearing glasses',
        'sunshine'], 'reverse_editing_direction': [False, True],
        'edit_guidance_scale': [10.0, 5.0]}
    return inputs

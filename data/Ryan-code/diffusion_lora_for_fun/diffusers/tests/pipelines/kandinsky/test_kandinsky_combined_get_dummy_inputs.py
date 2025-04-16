def get_dummy_inputs(self, device, seed=0):
    prior_dummy = PriorDummies()
    dummy = InpaintDummies()
    inputs = prior_dummy.get_dummy_inputs(device=device, seed=seed)
    inputs.update(dummy.get_dummy_inputs(device=device, seed=seed))
    inputs.pop('image_embeds')
    inputs.pop('negative_image_embeds')
    return inputs

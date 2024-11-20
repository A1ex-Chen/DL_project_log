def get_dummy_inputs(self, device, height=64, width=64, seed=0):
    inputs = super().get_dummy_inputs(device, seed, height=height, width=
        width, num_images=2)
    inputs['adapter_conditioning_scale'] = [0.5, 0.5]
    return inputs

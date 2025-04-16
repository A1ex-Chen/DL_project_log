def get_dummy_inputs_pipe(self, device, seed=0):
    inputs = self.get_dummy_inputs(device, seed=seed)
    inputs['output_type'] = 'np'
    inputs['return_dict'] = False
    return inputs

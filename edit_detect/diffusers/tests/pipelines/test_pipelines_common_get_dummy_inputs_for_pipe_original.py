def get_dummy_inputs_for_pipe_original(self, device, seed=0):
    inputs = {}
    for k, v in self.get_dummy_inputs_pipe(device, seed=seed).items():
        if k in set(inspect.signature(self.original_pipeline_class.__call__
            ).parameters.keys()):
            inputs[k] = v
    return inputs

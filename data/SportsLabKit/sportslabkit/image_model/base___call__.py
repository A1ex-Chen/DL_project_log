def __call__(self, inputs, **kwargs):
    inputs = self._check_and_fix_inputs(inputs)
    results = self.forward(inputs, **kwargs)
    embeddings = self._check_and_fix_outputs(results)
    return embeddings

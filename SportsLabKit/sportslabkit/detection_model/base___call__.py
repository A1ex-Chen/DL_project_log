def __call__(self, inputs, **kwargs):
    inputs = self._check_and_fix_inputs(inputs)
    results = self.forward(inputs, **kwargs)
    results = self._check_and_fix_outputs(results, inputs)
    detections = self._postprocess(results)
    return detections

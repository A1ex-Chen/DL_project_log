def test_pipeline_call_signature(self):
    self.assertTrue(hasattr(self.pipeline_class, '__call__'),
        f'{self.pipeline_class} should have a `__call__` method')
    parameters = inspect.signature(self.pipeline_class.__call__).parameters
    optional_parameters = set()
    for k, v in parameters.items():
        if v.default != inspect._empty:
            optional_parameters.add(k)
    parameters = set(parameters.keys())
    parameters.remove('self')
    parameters.discard('kwargs')
    remaining_required_parameters = set()
    for param in self.params:
        if param not in parameters:
            remaining_required_parameters.add(param)
    self.assertTrue(len(remaining_required_parameters) == 0,
        f'Required parameters not present: {remaining_required_parameters}')
    remaining_required_optional_parameters = set()
    for param in self.required_optional_params:
        if param not in optional_parameters:
            remaining_required_optional_parameters.add(param)
    self.assertTrue(len(remaining_required_optional_parameters) == 0,
        f'Required optional parameters not present: {remaining_required_optional_parameters}'
        )

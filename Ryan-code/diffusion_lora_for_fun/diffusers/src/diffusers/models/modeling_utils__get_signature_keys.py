@classmethod
def _get_signature_keys(cls, obj):
    parameters = inspect.signature(obj.__init__).parameters
    required_parameters = {k: v for k, v in parameters.items() if v.default ==
        inspect._empty}
    optional_parameters = set({k for k, v in parameters.items() if v.
        default != inspect._empty})
    expected_modules = set(required_parameters.keys()) - {'self'}
    return expected_modules, optional_parameters

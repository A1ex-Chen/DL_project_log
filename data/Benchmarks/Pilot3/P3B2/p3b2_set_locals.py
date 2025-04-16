def set_locals(self):
    """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries               describing the additional parameters for the
        benchmark.
        """
    if required is not None:
        self.required = set(required)
    if additional_definitions is not None:
        self.additional_definitions = additional_definitions

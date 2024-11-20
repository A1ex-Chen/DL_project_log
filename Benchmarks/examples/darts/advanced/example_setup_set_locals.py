def set_locals(self):
    """Set parameters for the benchmark.

        Args:
            required: set of required parameters for the benchmark.
        """
    if REQUIRED is not None:
        self.required = set(REQUIRED)
    if additional_definitions is not None:
        self.additional_definitions = additional_definitions

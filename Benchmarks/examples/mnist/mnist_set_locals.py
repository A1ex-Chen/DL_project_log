def set_locals(self):
    if required is not None:
        self.required = set(required)
    if additional_definitions is not None:
        self.additional_definitions = additional_definitions

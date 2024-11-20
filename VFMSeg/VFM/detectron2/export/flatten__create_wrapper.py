def _create_wrapper(self, traced_model):
    """
        Return a function that has an input/output interface the same as the
        original model, but it calls the given traced model under the hood.
        """

    def forward(*args):
        flattened_inputs, _ = flatten_to_tuple(args)
        flattened_outputs = traced_model(*flattened_inputs)
        return self.outputs_schema(flattened_outputs)
    return forward

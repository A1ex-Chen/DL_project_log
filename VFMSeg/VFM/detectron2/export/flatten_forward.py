def forward(*args):
    flattened_inputs, _ = flatten_to_tuple(args)
    flattened_outputs = traced_model(*flattened_inputs)
    return self.outputs_schema(flattened_outputs)

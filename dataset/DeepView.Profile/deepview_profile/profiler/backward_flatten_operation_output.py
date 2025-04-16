def flatten_operation_output(operation_output):
    if isinstance(operation_output, torch.Tensor):
        return [operation_output]
    elif not isinstance(operation_output, tuple) and not isinstance(
        operation_output, list):
        return []
    flattened = []
    for value in operation_output:
        flattened.extend(flatten_operation_output(value))
    return flattened

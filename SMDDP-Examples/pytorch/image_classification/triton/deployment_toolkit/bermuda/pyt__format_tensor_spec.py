def _format_tensor_spec(tensor_spec):
    tensor_spec = tensor_spec._replace(shape=list(tensor_spec.shape))
    tensor_spec = dict(tensor_spec._asdict())
    return tensor_spec

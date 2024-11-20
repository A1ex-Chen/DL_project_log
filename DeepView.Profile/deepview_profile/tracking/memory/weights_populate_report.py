def populate_report(self, builder):
    for param, (name, stack) in self._module_parameters.items():
        if not param.is_cuda:
            continue
        builder.add_weight_entry(weight_name=name, size_bytes=
            tensor_size_bytes(param), grad_size_bytes=tensor_size_bytes(
            param.grad), stack_context=stack)

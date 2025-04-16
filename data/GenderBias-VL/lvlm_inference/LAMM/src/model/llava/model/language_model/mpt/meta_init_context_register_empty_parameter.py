def register_empty_parameter(module, name, param):
    old_register_parameter(module, name, param)
    if param is not None:
        param_cls = type(module._parameters[name])
        kwargs = module._parameters[name].__dict__
        module._parameters[name] = param_cls(module._parameters[name].to(
            device), **kwargs)

def hook(*args, **kwargs):
    name = args[1]
    parameter = args[2]
    retval = func(*args, **kwargs)
    if parameter is not None and parameter not in self._module_parameters:
        self._module_parameters[parameter] = name, CallStack.from_here(self
            ._project_root, start_from=2)
    return retval

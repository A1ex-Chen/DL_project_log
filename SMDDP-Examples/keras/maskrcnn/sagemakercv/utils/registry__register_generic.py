def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

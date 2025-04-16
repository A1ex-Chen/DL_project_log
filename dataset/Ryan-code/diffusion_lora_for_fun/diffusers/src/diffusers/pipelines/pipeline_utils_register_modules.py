def register_modules(self, **kwargs):
    for name, module in kwargs.items():
        if module is None or isinstance(module, (tuple, list)) and module[0
            ] is None:
            register_dict = {name: (None, None)}
        else:
            library, class_name = _fetch_class_library_tuple(module)
            register_dict = {name: (library, class_name)}
        self.register_to_config(**register_dict)
        setattr(self, name, module)

def __setattr__(self, name: str, value: Any):
    if name in self.__dict__ and hasattr(self.config, name):
        if isinstance(getattr(self.config, name), (tuple, list)):
            if value is not None and self.config[name][0] is not None:
                class_library_tuple = _fetch_class_library_tuple(value)
            else:
                class_library_tuple = None, None
            self.register_to_config(**{name: class_library_tuple})
        else:
            self.register_to_config(**{name: value})
    super().__setattr__(name, value)

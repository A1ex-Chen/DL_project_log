def for_model(self, model):
    for name, module in model.named_modules():
        self._module_names_by_id[id(module)] = name
    return self

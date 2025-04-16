def set(self, name, _type, output=None, pos=None, var=None):
    if var is not None:
        self.attn_variables[name] = var
    elif name in self.duplication_dict:
        assert self.duplication_dict[name
            ] in self.attn_variables, 'Duplication variable {} is not initialized yet.'.format(
            name)
        self.attn_variables[name] = self.attn_variables[self.
            duplication_dict[name]].copy()
    else:
        var = Variable(output, name, _type, pos)
        self.attn_variables[name] = var

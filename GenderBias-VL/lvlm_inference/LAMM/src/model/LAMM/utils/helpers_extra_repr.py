def extra_repr(self) ->str:
    named_modules = set()
    for p in self.named_modules():
        named_modules.update([p[0]])
    named_modules = list(named_modules)
    string_repr = ''
    for p in self.named_parameters():
        name = p[0].split('.')[0]
        if name not in named_modules:
            string_repr += self.get_readable_tensor_repr(name, p)
    for p in self.named_buffers():
        name = p[0].split('.')[0]
        string_repr += self.get_readable_tensor_repr(name, p)
    return string_repr

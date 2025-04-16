def _module_list(module, flatten_sequential=False):
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined),
                    child_module))
        else:
            ml.append((name, name, module))
    return ml

def functionalized(self, *user_args):
    out = Graphed.apply(*(user_args + module_params))
    return out[0] if outputs_was_tensor else out

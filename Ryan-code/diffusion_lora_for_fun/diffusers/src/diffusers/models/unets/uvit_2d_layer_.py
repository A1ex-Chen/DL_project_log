def layer_(*args):
    return checkpoint(layer, *args)

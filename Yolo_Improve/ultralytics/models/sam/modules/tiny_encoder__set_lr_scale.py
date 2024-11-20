def _set_lr_scale(m, scale):
    """Sets the learning rate scale for each layer in the model based on the layer's depth."""
    for p in m.parameters():
        p.lr_scale = scale

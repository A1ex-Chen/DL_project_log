def _check_lr_scale(m):
    """Checks if the learning rate scale attribute is present in module's parameters."""
    for p in m.parameters():
        assert hasattr(p, 'lr_scale'), p.param_name

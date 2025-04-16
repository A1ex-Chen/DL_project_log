def do_ema_update(source_model, shadow_models, decay_consts):
    """Performs the exponential model average (EMA) update.

    Args:
        source_model:   The source model.
        shadow_models:  A list of shadow models to be updated.
        decay_consts:   A list of EMA decay constants
                        corresponding to the shadow models.
    """
    assert len(shadow_models) == len(decay_consts)
    model_params = OrderedDict(source_model.named_parameters())
    model_buffers = OrderedDict(source_model.named_buffers())
    for shadow_model, ema_decay in zip(shadow_models, decay_consts):
        shadow_params = OrderedDict(shadow_model.named_parameters())
        shadow_buffers = OrderedDict(shadow_model.named_buffers())
        assert ema_decay <= 1 and ema_decay >= 0
        assert model_params.keys() == shadow_params.keys()
        assert model_buffers.keys() == shadow_buffers.keys()
        for name, param in model_params.items():
            shadow_params[name].add_((1.0 - ema_decay) * (param -
                shadow_params[name]))
        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

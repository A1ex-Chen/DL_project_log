def loss_fn(outputs, targets, **kwargs):
    assert kwargs['loss_kwarg'] is None
    return loss(outputs, targets)

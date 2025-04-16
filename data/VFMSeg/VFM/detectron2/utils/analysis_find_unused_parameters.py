def find_unused_parameters(model: nn.Module, inputs: Any) ->List[str]:
    """
    Given a model, find parameters that do not contribute
    to the loss.

    Args:
        model: a model in training mode that returns losses
        inputs: argument or a tuple of arguments. Inputs of the model

    Returns:
        list[str]: the name of unused parameters
    """
    assert model.training
    for _, prm in model.named_parameters():
        prm.grad = None
    if isinstance(inputs, tuple):
        losses = model(*inputs)
    else:
        losses = model(inputs)
    if isinstance(losses, dict):
        losses = sum(losses.values())
    losses.backward()
    unused: List[str] = []
    for name, prm in model.named_parameters():
        if prm.grad is None:
            unused.append(name)
        prm.grad = None
    return unused

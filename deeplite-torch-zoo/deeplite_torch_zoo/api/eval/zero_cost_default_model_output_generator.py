def default_model_output_generator(model, dataloader=None, shuffle_data=
    True, input_gradient=False, device='cpu'):
    loss_kwargs = {}
    try:
        loader = dataloader if shuffle_data else repeat(next(iter(dataloader)))
    except StopIteration:
        return
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs.requires_grad_(input_gradient)
        outputs = model(inputs)
        yield inputs, outputs, targets, loss_kwargs

@ZERO_COST_SCORES.register('zico')
def zico(model, model_output_generator, loss_fn, n_steps=2, mode='sum',
    reduction='sum'):
    grad_dict = {}
    data_generator = model_output_generator(model)
    for step in range(n_steps):
        model.zero_grad()
        _, outputs, targets, loss_kwargs = next(data_generator)
        loss = loss_fn(outputs, targets, **loss_kwargs)
        loss.backward()
        grad_dict = get_grad(model, grad_dict, step)
    return aggregate_statistic(compute_zico(grad_dict, mode=mode),
        reduction=reduction)

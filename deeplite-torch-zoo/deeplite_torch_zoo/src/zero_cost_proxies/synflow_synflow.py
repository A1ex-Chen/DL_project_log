@ZERO_COST_SCORES.register('synflow')
def synflow(model, model_output_generator, loss_fn=None, dummify_bns=True,
    bn_training_mode=False, reduction='sum', output_post_processing=None):
    if output_post_processing is None:
        output_post_processing = lambda tensors: torch.cat([x.flatten() for
            x in tensors])
    if dummify_bns:
        model.apply(dummify_bns_fn)

    @torch.no_grad()
    def linearize(model):
        for param in model.state_dict().values():
            param.abs_()
    model.double()
    if not bn_training_mode:
        if not dummify_bns:
            model.eval()
        linearize(model)
    inp, _, _, _ = next(model_output_generator(nn.Identity()))
    inputs = torch.ones((1, *inp.shape[1:]), device=inp.device, dtype=torch
        .float64)
    outputs = model(inputs)
    torch.sum(output_post_processing(outputs)).backward()
    grads_abs = get_layerwise_metric_values(model, get_synflow)
    return aggregate_statistic(grads_abs, reduction=reduction)

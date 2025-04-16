@ZERO_COST_SCORES.register('l2_norm')
def get_l2_norm_array(model, model_output_generator=None, loss_fn=None,
    reduction='sum'):
    norm = get_layerwise_metric_values(model, lambda l: l.weight.norm())
    return aggregate_statistic(norm, reduction=reduction)

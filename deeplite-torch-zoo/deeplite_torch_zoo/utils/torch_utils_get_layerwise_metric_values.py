def get_layerwise_metric_values(model, metric_fn, target_layer_types=None):
    metric_array = []
    target_layer_types = target_layer_types or TRAINABLE_LAYERS
    for layer in model.modules():
        if isinstance(layer, target_layer_types):
            metric_array.append(metric_fn(layer))
    return metric_array

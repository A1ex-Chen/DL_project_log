def rescale(param, layer_id):
    param.div_(math.sqrt(2.0 * layer_id))

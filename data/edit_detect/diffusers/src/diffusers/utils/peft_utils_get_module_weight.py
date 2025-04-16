def get_module_weight(weight_for_adapter, module_name):
    if not isinstance(weight_for_adapter, dict):
        return weight_for_adapter
    for layer_name, weight_ in weight_for_adapter.items():
        if layer_name in module_name:
            return weight_
    parts = module_name.split('.')
    key = f'{parts[0]}.{parts[1]}.attentions.{parts[3]}'
    block_weight = weight_for_adapter.get(key, 1.0)
    return block_weight

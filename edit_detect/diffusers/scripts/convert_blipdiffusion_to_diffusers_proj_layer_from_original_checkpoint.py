def proj_layer_from_original_checkpoint(model, diffuser_proj_prefix,
    original_proj_prefix):
    proj_layer = {}
    proj_layer.update({f'{diffuser_proj_prefix}.dense1.weight': model[
        f'{original_proj_prefix}.dense1.weight']})
    proj_layer.update({f'{diffuser_proj_prefix}.dense1.bias': model[
        f'{original_proj_prefix}.dense1.bias']})
    proj_layer.update({f'{diffuser_proj_prefix}.dense2.weight': model[
        f'{original_proj_prefix}.dense2.weight']})
    proj_layer.update({f'{diffuser_proj_prefix}.dense2.bias': model[
        f'{original_proj_prefix}.dense2.bias']})
    proj_layer.update({f'{diffuser_proj_prefix}.LayerNorm.weight': model[
        f'{original_proj_prefix}.LayerNorm.weight']})
    proj_layer.update({f'{diffuser_proj_prefix}.LayerNorm.bias': model[
        f'{original_proj_prefix}.LayerNorm.bias']})
    return proj_layer

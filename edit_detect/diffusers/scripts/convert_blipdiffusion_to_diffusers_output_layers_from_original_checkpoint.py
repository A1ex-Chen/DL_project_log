def output_layers_from_original_checkpoint(model, diffuser_output_prefix,
    original_output_prefix):
    output_layers = {}
    output_layers.update({f'{diffuser_output_prefix}.dense.weight': model[
        f'{original_output_prefix}.dense.weight']})
    output_layers.update({f'{diffuser_output_prefix}.dense.bias': model[
        f'{original_output_prefix}.dense.bias']})
    output_layers.update({f'{diffuser_output_prefix}.LayerNorm.weight':
        model[f'{original_output_prefix}.LayerNorm.weight']})
    output_layers.update({f'{diffuser_output_prefix}.LayerNorm.bias': model
        [f'{original_output_prefix}.LayerNorm.bias']})
    return output_layers

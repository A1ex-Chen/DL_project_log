def set_attention_weights(new_layer, old_checkpoint, index):
    new_layer.query.weight.data = old_checkpoint[f'all_modules.{index}.NIN_0.W'
        ].data.T
    new_layer.key.weight.data = old_checkpoint[f'all_modules.{index}.NIN_1.W'
        ].data.T
    new_layer.value.weight.data = old_checkpoint[f'all_modules.{index}.NIN_2.W'
        ].data.T
    new_layer.query.bias.data = old_checkpoint[f'all_modules.{index}.NIN_0.b'
        ].data
    new_layer.key.bias.data = old_checkpoint[f'all_modules.{index}.NIN_1.b'
        ].data
    new_layer.value.bias.data = old_checkpoint[f'all_modules.{index}.NIN_2.b'
        ].data
    new_layer.proj_attn.weight.data = old_checkpoint[
        f'all_modules.{index}.NIN_3.W'].data.T
    new_layer.proj_attn.bias.data = old_checkpoint[
        f'all_modules.{index}.NIN_3.b'].data
    new_layer.group_norm.weight.data = old_checkpoint[
        f'all_modules.{index}.GroupNorm_0.weight'].data
    new_layer.group_norm.bias.data = old_checkpoint[
        f'all_modules.{index}.GroupNorm_0.bias'].data

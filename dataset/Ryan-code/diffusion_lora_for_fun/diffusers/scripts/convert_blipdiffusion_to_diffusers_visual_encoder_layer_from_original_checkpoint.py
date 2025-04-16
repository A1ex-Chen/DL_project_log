def visual_encoder_layer_from_original_checkpoint(model, diffuser_prefix,
    original_prefix):
    visual_encoder_layer = {}
    visual_encoder_layer.update({f'{diffuser_prefix}.layer_norm1.weight':
        model[f'{original_prefix}.ln_1.weight']})
    visual_encoder_layer.update({f'{diffuser_prefix}.layer_norm1.bias':
        model[f'{original_prefix}.ln_1.bias']})
    visual_encoder_layer.update({f'{diffuser_prefix}.layer_norm2.weight':
        model[f'{original_prefix}.ln_2.weight']})
    visual_encoder_layer.update({f'{diffuser_prefix}.layer_norm2.bias':
        model[f'{original_prefix}.ln_2.bias']})
    visual_encoder_layer.update({f'{diffuser_prefix}.self_attn.qkv.weight':
        model[f'{original_prefix}.attn.in_proj_weight']})
    visual_encoder_layer.update({f'{diffuser_prefix}.self_attn.qkv.bias':
        model[f'{original_prefix}.attn.in_proj_bias']})
    visual_encoder_layer.update({
        f'{diffuser_prefix}.self_attn.projection.weight': model[
        f'{original_prefix}.attn.out_proj.weight']})
    visual_encoder_layer.update({
        f'{diffuser_prefix}.self_attn.projection.bias': model[
        f'{original_prefix}.attn.out_proj.bias']})
    visual_encoder_layer.update({f'{diffuser_prefix}.mlp.fc1.weight': model
        [f'{original_prefix}.mlp.c_fc.weight']})
    visual_encoder_layer.update({f'{diffuser_prefix}.mlp.fc1.bias': model[
        f'{original_prefix}.mlp.c_fc.bias']})
    visual_encoder_layer.update({f'{diffuser_prefix}.mlp.fc2.weight': model
        [f'{original_prefix}.mlp.c_proj.weight']})
    visual_encoder_layer.update({f'{diffuser_prefix}.mlp.fc2.bias': model[
        f'{original_prefix}.mlp.c_proj.bias']})
    return visual_encoder_layer

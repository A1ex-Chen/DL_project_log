def visual_encoder_from_original_checkpoint(model, diffuser_prefix,
    original_prefix):
    visual_encoder = {}
    visual_encoder.update({f'{diffuser_prefix}.embeddings.class_embedding':
        model[f'{original_prefix}.class_embedding'].unsqueeze(0).unsqueeze(0)})
    visual_encoder.update({
        f'{diffuser_prefix}.embeddings.position_embedding': model[
        f'{original_prefix}.positional_embedding'].unsqueeze(0)})
    visual_encoder.update({
        f'{diffuser_prefix}.embeddings.patch_embedding.weight': model[
        f'{original_prefix}.conv1.weight']})
    visual_encoder.update({f'{diffuser_prefix}.pre_layernorm.weight': model
        [f'{original_prefix}.ln_pre.weight']})
    visual_encoder.update({f'{diffuser_prefix}.pre_layernorm.bias': model[
        f'{original_prefix}.ln_pre.bias']})
    for i in range(blip2config.vision_config.num_hidden_layers):
        visual_encoder.update(visual_encoder_layer_from_original_checkpoint
            (model, f'{diffuser_prefix}.encoder.layers.{i}',
            f'{original_prefix}.transformer.resblocks.{i}'))
    visual_encoder.update({f'{diffuser_prefix}.post_layernorm.weight':
        model['blip.ln_vision.weight']})
    visual_encoder.update({f'{diffuser_prefix}.post_layernorm.bias': model[
        'blip.ln_vision.bias']})
    return visual_encoder

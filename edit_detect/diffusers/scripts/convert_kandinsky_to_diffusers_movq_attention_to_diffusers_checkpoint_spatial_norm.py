def movq_attention_to_diffusers_checkpoint_spatial_norm(checkpoint, *,
    diffusers_attention_prefix, attention_prefix):
    return {f'{diffusers_attention_prefix}.spatial_norm.norm_layer.weight':
        checkpoint[f'{attention_prefix}.norm.norm_layer.weight'],
        f'{diffusers_attention_prefix}.spatial_norm.norm_layer.bias':
        checkpoint[f'{attention_prefix}.norm.norm_layer.bias'],
        f'{diffusers_attention_prefix}.spatial_norm.conv_y.weight':
        checkpoint[f'{attention_prefix}.norm.conv_y.weight'],
        f'{diffusers_attention_prefix}.spatial_norm.conv_y.bias':
        checkpoint[f'{attention_prefix}.norm.conv_y.bias'],
        f'{diffusers_attention_prefix}.spatial_norm.conv_b.weight':
        checkpoint[f'{attention_prefix}.norm.conv_b.weight'],
        f'{diffusers_attention_prefix}.spatial_norm.conv_b.bias':
        checkpoint[f'{attention_prefix}.norm.conv_b.bias'],
        f'{diffusers_attention_prefix}.to_q.weight': checkpoint[
        f'{attention_prefix}.q.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.to_q.bias': checkpoint[
        f'{attention_prefix}.q.bias'],
        f'{diffusers_attention_prefix}.to_k.weight': checkpoint[
        f'{attention_prefix}.k.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.to_k.bias': checkpoint[
        f'{attention_prefix}.k.bias'],
        f'{diffusers_attention_prefix}.to_v.weight': checkpoint[
        f'{attention_prefix}.v.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.to_v.bias': checkpoint[
        f'{attention_prefix}.v.bias'],
        f'{diffusers_attention_prefix}.to_out.0.weight': checkpoint[
        f'{attention_prefix}.proj_out.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.to_out.0.bias': checkpoint[
        f'{attention_prefix}.proj_out.bias']}

def attention_to_diffusers_checkpoint(checkpoint, *,
    diffusers_attention_prefix, attention_prefix, num_head_channels):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({
        f'{diffusers_attention_prefix}.group_norm.weight': checkpoint[
        f'{attention_prefix}.norm.weight'],
        f'{diffusers_attention_prefix}.group_norm.bias': checkpoint[
        f'{attention_prefix}.norm.bias']})
    [q_weight, k_weight, v_weight], [q_bias, k_bias, v_bias
        ] = split_attentions(weight=checkpoint[
        f'{attention_prefix}.qkv.weight'][:, :, 0], bias=checkpoint[
        f'{attention_prefix}.qkv.bias'], split=3, chunk_size=num_head_channels)
    diffusers_checkpoint.update({
        f'{diffusers_attention_prefix}.to_q.weight': q_weight,
        f'{diffusers_attention_prefix}.to_q.bias': q_bias,
        f'{diffusers_attention_prefix}.to_k.weight': k_weight,
        f'{diffusers_attention_prefix}.to_k.bias': k_bias,
        f'{diffusers_attention_prefix}.to_v.weight': v_weight,
        f'{diffusers_attention_prefix}.to_v.bias': v_bias})
    [encoder_k_weight, encoder_v_weight], [encoder_k_bias, encoder_v_bias
        ] = split_attentions(weight=checkpoint[
        f'{attention_prefix}.encoder_kv.weight'][:, :, 0], bias=checkpoint[
        f'{attention_prefix}.encoder_kv.bias'], split=2, chunk_size=
        num_head_channels)
    diffusers_checkpoint.update({
        f'{diffusers_attention_prefix}.add_k_proj.weight': encoder_k_weight,
        f'{diffusers_attention_prefix}.add_k_proj.bias': encoder_k_bias,
        f'{diffusers_attention_prefix}.add_v_proj.weight': encoder_v_weight,
        f'{diffusers_attention_prefix}.add_v_proj.bias': encoder_v_bias})
    diffusers_checkpoint.update({
        f'{diffusers_attention_prefix}.to_out.0.weight': checkpoint[
        f'{attention_prefix}.proj_out.weight'][:, :, 0],
        f'{diffusers_attention_prefix}.to_out.0.bias': checkpoint[
        f'{attention_prefix}.proj_out.bias']})
    return diffusers_checkpoint

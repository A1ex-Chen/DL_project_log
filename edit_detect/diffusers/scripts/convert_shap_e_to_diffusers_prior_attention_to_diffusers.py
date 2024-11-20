def prior_attention_to_diffusers(checkpoint, *, diffusers_attention_prefix,
    original_attention_prefix, attention_head_dim):
    diffusers_checkpoint = {}
    [q_weight, k_weight, v_weight], [q_bias, k_bias, v_bias
        ] = split_attentions(weight=checkpoint[
        f'{original_attention_prefix}.c_qkv.weight'], bias=checkpoint[
        f'{original_attention_prefix}.c_qkv.bias'], split=3, chunk_size=
        attention_head_dim)
    diffusers_checkpoint.update({
        f'{diffusers_attention_prefix}.to_q.weight': q_weight,
        f'{diffusers_attention_prefix}.to_q.bias': q_bias,
        f'{diffusers_attention_prefix}.to_k.weight': k_weight,
        f'{diffusers_attention_prefix}.to_k.bias': k_bias,
        f'{diffusers_attention_prefix}.to_v.weight': v_weight,
        f'{diffusers_attention_prefix}.to_v.bias': v_bias})
    diffusers_checkpoint.update({
        f'{diffusers_attention_prefix}.to_out.0.weight': checkpoint[
        f'{original_attention_prefix}.c_proj.weight'],
        f'{diffusers_attention_prefix}.to_out.0.bias': checkpoint[
        f'{original_attention_prefix}.c_proj.bias']})
    return diffusers_checkpoint

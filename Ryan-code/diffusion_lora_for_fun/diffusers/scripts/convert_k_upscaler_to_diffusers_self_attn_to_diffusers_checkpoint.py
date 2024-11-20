def self_attn_to_diffusers_checkpoint(checkpoint, *,
    diffusers_attention_prefix, attention_prefix):
    weight_q, weight_k, weight_v = checkpoint[
        f'{attention_prefix}.qkv_proj.weight'].chunk(3, dim=0)
    bias_q, bias_k, bias_v = checkpoint[f'{attention_prefix}.qkv_proj.bias'
        ].chunk(3, dim=0)
    rv = {f'{diffusers_attention_prefix}.norm1.linear.weight': checkpoint[
        f'{attention_prefix}.norm_in.mapper.weight'],
        f'{diffusers_attention_prefix}.norm1.linear.bias': checkpoint[
        f'{attention_prefix}.norm_in.mapper.bias'],
        f'{diffusers_attention_prefix}.attn1.to_q.weight': weight_q.squeeze
        (-1).squeeze(-1), f'{diffusers_attention_prefix}.attn1.to_q.bias':
        bias_q, f'{diffusers_attention_prefix}.attn1.to_k.weight': weight_k
        .squeeze(-1).squeeze(-1),
        f'{diffusers_attention_prefix}.attn1.to_k.bias': bias_k,
        f'{diffusers_attention_prefix}.attn1.to_v.weight': weight_v.squeeze
        (-1).squeeze(-1), f'{diffusers_attention_prefix}.attn1.to_v.bias':
        bias_v, f'{diffusers_attention_prefix}.attn1.to_out.0.weight':
        checkpoint[f'{attention_prefix}.out_proj.weight'].squeeze(-1).
        squeeze(-1), f'{diffusers_attention_prefix}.attn1.to_out.0.bias':
        checkpoint[f'{attention_prefix}.out_proj.bias']}
    return rv

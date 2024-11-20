def cross_attn_to_diffusers_checkpoint(checkpoint, *,
    diffusers_attention_prefix, diffusers_attention_index, attention_prefix):
    weight_k, weight_v = checkpoint[f'{attention_prefix}.kv_proj.weight'
        ].chunk(2, dim=0)
    bias_k, bias_v = checkpoint[f'{attention_prefix}.kv_proj.bias'].chunk(2,
        dim=0)
    rv = {
        f'{diffusers_attention_prefix}.norm{diffusers_attention_index}.linear.weight'
        : checkpoint[f'{attention_prefix}.norm_dec.mapper.weight'],
        f'{diffusers_attention_prefix}.norm{diffusers_attention_index}.linear.bias'
        : checkpoint[f'{attention_prefix}.norm_dec.mapper.bias'],
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.norm_cross.weight'
        : checkpoint[f'{attention_prefix}.norm_enc.weight'],
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.norm_cross.bias'
        : checkpoint[f'{attention_prefix}.norm_enc.bias'],
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_q.weight'
        : checkpoint[f'{attention_prefix}.q_proj.weight'].squeeze(-1).
        squeeze(-1),
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_q.bias'
        : checkpoint[f'{attention_prefix}.q_proj.bias'],
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_k.weight'
        : weight_k.squeeze(-1).squeeze(-1),
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_k.bias'
        : bias_k,
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_v.weight'
        : weight_v.squeeze(-1).squeeze(-1),
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_v.bias'
        : bias_v,
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_out.0.weight'
        : checkpoint[f'{attention_prefix}.out_proj.weight'].squeeze(-1).
        squeeze(-1),
        f'{diffusers_attention_prefix}.attn{diffusers_attention_index}.to_out.0.bias'
        : checkpoint[f'{attention_prefix}.out_proj.bias']}
    return rv

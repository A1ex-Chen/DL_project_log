def transformer_attention_to_diffusers_checkpoint(checkpoint, *,
    diffusers_attention_prefix, attention_prefix):
    return {f'{diffusers_attention_prefix}.to_k.weight': checkpoint[
        f'{attention_prefix}.key.weight'],
        f'{diffusers_attention_prefix}.to_k.bias': checkpoint[
        f'{attention_prefix}.key.bias'],
        f'{diffusers_attention_prefix}.to_q.weight': checkpoint[
        f'{attention_prefix}.query.weight'],
        f'{diffusers_attention_prefix}.to_q.bias': checkpoint[
        f'{attention_prefix}.query.bias'],
        f'{diffusers_attention_prefix}.to_v.weight': checkpoint[
        f'{attention_prefix}.value.weight'],
        f'{diffusers_attention_prefix}.to_v.bias': checkpoint[
        f'{attention_prefix}.value.bias'],
        f'{diffusers_attention_prefix}.to_out.0.weight': checkpoint[
        f'{attention_prefix}.proj.weight'],
        f'{diffusers_attention_prefix}.to_out.0.bias': checkpoint[
        f'{attention_prefix}.proj.bias']}

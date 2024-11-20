def vqvae_attention_to_diffusers_checkpoint(checkpoint, *,
    diffusers_attention_prefix, attention_prefix):
    return {f'{diffusers_attention_prefix}.group_norm.weight': checkpoint[
        f'{attention_prefix}.norm.weight'],
        f'{diffusers_attention_prefix}.group_norm.bias': checkpoint[
        f'{attention_prefix}.norm.bias'],
        f'{diffusers_attention_prefix}.query.weight': checkpoint[
        f'{attention_prefix}.q.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.query.bias': checkpoint[
        f'{attention_prefix}.q.bias'],
        f'{diffusers_attention_prefix}.key.weight': checkpoint[
        f'{attention_prefix}.k.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.key.bias': checkpoint[
        f'{attention_prefix}.k.bias'],
        f'{diffusers_attention_prefix}.value.weight': checkpoint[
        f'{attention_prefix}.v.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.value.bias': checkpoint[
        f'{attention_prefix}.v.bias'],
        f'{diffusers_attention_prefix}.proj_attn.weight': checkpoint[
        f'{attention_prefix}.proj_out.weight'][:, :, 0, 0],
        f'{diffusers_attention_prefix}.proj_attn.bias': checkpoint[
        f'{attention_prefix}.proj_out.bias']}

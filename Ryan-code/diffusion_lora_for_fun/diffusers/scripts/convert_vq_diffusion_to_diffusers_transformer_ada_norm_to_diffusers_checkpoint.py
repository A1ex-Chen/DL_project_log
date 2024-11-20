def transformer_ada_norm_to_diffusers_checkpoint(checkpoint, *,
    diffusers_ada_norm_prefix, ada_norm_prefix):
    return {f'{diffusers_ada_norm_prefix}.emb.weight': checkpoint[
        f'{ada_norm_prefix}.emb.weight'],
        f'{diffusers_ada_norm_prefix}.linear.weight': checkpoint[
        f'{ada_norm_prefix}.linear.weight'],
        f'{diffusers_ada_norm_prefix}.linear.bias': checkpoint[
        f'{ada_norm_prefix}.linear.bias']}

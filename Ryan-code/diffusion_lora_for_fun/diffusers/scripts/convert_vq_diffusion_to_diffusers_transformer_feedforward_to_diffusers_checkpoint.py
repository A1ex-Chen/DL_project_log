def transformer_feedforward_to_diffusers_checkpoint(checkpoint, *,
    diffusers_feedforward_prefix, feedforward_prefix):
    return {f'{diffusers_feedforward_prefix}.net.0.proj.weight': checkpoint
        [f'{feedforward_prefix}.0.weight'],
        f'{diffusers_feedforward_prefix}.net.0.proj.bias': checkpoint[
        f'{feedforward_prefix}.0.bias'],
        f'{diffusers_feedforward_prefix}.net.2.weight': checkpoint[
        f'{feedforward_prefix}.2.weight'],
        f'{diffusers_feedforward_prefix}.net.2.bias': checkpoint[
        f'{feedforward_prefix}.2.bias']}

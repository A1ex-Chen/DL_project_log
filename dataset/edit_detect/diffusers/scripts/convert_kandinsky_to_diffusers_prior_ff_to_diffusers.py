def prior_ff_to_diffusers(checkpoint, *, diffusers_ff_prefix,
    original_ff_prefix):
    diffusers_checkpoint = {f'{diffusers_ff_prefix}.net.{0}.proj.weight':
        checkpoint[f'{original_ff_prefix}.c_fc.weight'],
        f'{diffusers_ff_prefix}.net.{0}.proj.bias': checkpoint[
        f'{original_ff_prefix}.c_fc.bias'],
        f'{diffusers_ff_prefix}.net.{2}.weight': checkpoint[
        f'{original_ff_prefix}.c_proj.weight'],
        f'{diffusers_ff_prefix}.net.{2}.bias': checkpoint[
        f'{original_ff_prefix}.c_proj.bias']}
    return diffusers_checkpoint

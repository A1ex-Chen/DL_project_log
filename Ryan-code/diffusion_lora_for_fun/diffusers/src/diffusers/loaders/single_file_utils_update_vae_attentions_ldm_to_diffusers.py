def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint,
    mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping['old'], mapping['new']
            ).replace('norm.weight', 'group_norm.weight').replace('norm.bias',
            'group_norm.bias').replace('q.weight', 'to_q.weight').replace(
            'q.bias', 'to_q.bias').replace('k.weight', 'to_k.weight').replace(
            'k.bias', 'to_k.bias').replace('v.weight', 'to_v.weight').replace(
            'v.bias', 'to_v.bias').replace('proj_out.weight', 'to_out.0.weight'
            ).replace('proj_out.bias', 'to_out.0.bias')
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)
        shape = new_checkpoint[diffusers_key].shape
        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:,
                :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:,
                :, 0, 0]

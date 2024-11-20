def unet_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_in.weight': checkpoint[
        'inner_model.proj_in.weight'], 'conv_in.bias': checkpoint[
        'inner_model.proj_in.bias']})
    diffusers_checkpoint.update({'time_proj.weight': checkpoint[
        'inner_model.timestep_embed.weight'].squeeze(-1),
        'time_embedding.linear_1.weight': checkpoint[
        'inner_model.mapping.0.weight'], 'time_embedding.linear_1.bias':
        checkpoint['inner_model.mapping.0.bias'],
        'time_embedding.linear_2.weight': checkpoint[
        'inner_model.mapping.2.weight'], 'time_embedding.linear_2.bias':
        checkpoint['inner_model.mapping.2.bias'],
        'time_embedding.cond_proj.weight': checkpoint[
        'inner_model.mapping_cond.weight']})
    for down_block_idx, down_block in enumerate(model.down_blocks):
        diffusers_checkpoint.update(block_to_diffusers_checkpoint(
            down_block, checkpoint, down_block_idx, 'down'))
    for up_block_idx, up_block in enumerate(model.up_blocks):
        diffusers_checkpoint.update(block_to_diffusers_checkpoint(up_block,
            checkpoint, up_block_idx, 'up'))
    diffusers_checkpoint.update({'conv_out.weight': checkpoint[
        'inner_model.proj_out.weight'], 'conv_out.bias': checkpoint[
        'inner_model.proj_out.bias']})
    return diffusers_checkpoint

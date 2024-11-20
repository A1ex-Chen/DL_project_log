def unet_midblock_to_diffusers_checkpoint(model, checkpoint, *,
    num_head_channels):
    diffusers_checkpoint = {}
    original_block_idx = 0
    diffusers_checkpoint.update(resnet_to_diffusers_checkpoint(checkpoint,
        diffusers_resnet_prefix='mid_block.resnets.0', resnet_prefix=
        f'middle_block.{original_block_idx}'))
    original_block_idx += 1
    if hasattr(model.mid_block, 'attentions') and model.mid_block.attentions[0
        ] is not None:
        diffusers_checkpoint.update(attention_to_diffusers_checkpoint(
            checkpoint, diffusers_attention_prefix='mid_block.attentions.0',
            attention_prefix=f'middle_block.{original_block_idx}',
            num_head_channels=num_head_channels))
        original_block_idx += 1
    diffusers_checkpoint.update(resnet_to_diffusers_checkpoint(checkpoint,
        diffusers_resnet_prefix='mid_block.resnets.1', resnet_prefix=
        f'middle_block.{original_block_idx}'))
    return diffusers_checkpoint

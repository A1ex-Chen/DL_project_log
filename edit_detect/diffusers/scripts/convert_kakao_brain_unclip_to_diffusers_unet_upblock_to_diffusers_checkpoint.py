def unet_upblock_to_diffusers_checkpoint(model, checkpoint, *,
    diffusers_up_block_idx, original_up_block_idx, original_unet_prefix,
    num_head_channels):
    diffusers_checkpoint = {}
    diffusers_resnet_prefix = f'up_blocks.{diffusers_up_block_idx}.resnets'
    original_up_block_prefix = f'{original_unet_prefix}.output_blocks'
    up_block = model.up_blocks[diffusers_up_block_idx]
    num_resnets = len(up_block.resnets)
    if up_block.upsamplers is None:
        upsampler = False
    else:
        assert len(up_block.upsamplers) == 1
        upsampler = True
        num_resnets += 1
    has_attentions = hasattr(up_block, 'attentions')
    for resnet_idx_inc in range(num_resnets):
        if upsampler and resnet_idx_inc == num_resnets - 1:
            if has_attentions:
                original_resnet_block_idx = 2
            else:
                original_resnet_block_idx = 1
            full_resnet_prefix = (
                f'{original_up_block_prefix}.{original_up_block_idx + resnet_idx_inc - 1}.{original_resnet_block_idx}'
                )
            full_diffusers_resnet_prefix = (
                f'up_blocks.{diffusers_up_block_idx}.upsamplers.0')
        else:
            full_resnet_prefix = (
                f'{original_up_block_prefix}.{original_up_block_idx + resnet_idx_inc}.0'
                )
            full_diffusers_resnet_prefix = (
                f'{diffusers_resnet_prefix}.{resnet_idx_inc}')
        diffusers_checkpoint.update(resnet_to_diffusers_checkpoint(
            checkpoint, resnet_prefix=full_resnet_prefix,
            diffusers_resnet_prefix=full_diffusers_resnet_prefix))
    if has_attentions:
        num_attentions = len(up_block.attentions)
        diffusers_attention_prefix = (
            f'up_blocks.{diffusers_up_block_idx}.attentions')
        for attention_idx_inc in range(num_attentions):
            full_attention_prefix = (
                f'{original_up_block_prefix}.{original_up_block_idx + attention_idx_inc}.1'
                )
            full_diffusers_attention_prefix = (
                f'{diffusers_attention_prefix}.{attention_idx_inc}')
            diffusers_checkpoint.update(attention_to_diffusers_checkpoint(
                checkpoint, attention_prefix=full_attention_prefix,
                diffusers_attention_prefix=full_diffusers_attention_prefix,
                num_head_channels=num_head_channels))
    num_original_down_blocks = num_resnets - 1 if upsampler else num_resnets
    return diffusers_checkpoint, num_original_down_blocks

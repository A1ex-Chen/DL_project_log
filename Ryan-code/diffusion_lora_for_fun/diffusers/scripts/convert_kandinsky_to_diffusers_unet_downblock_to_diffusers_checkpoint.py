def unet_downblock_to_diffusers_checkpoint(model, checkpoint, *,
    diffusers_down_block_idx, original_down_block_idx, num_head_channels):
    diffusers_checkpoint = {}
    diffusers_resnet_prefix = f'down_blocks.{diffusers_down_block_idx}.resnets'
    original_down_block_prefix = 'input_blocks'
    down_block = model.down_blocks[diffusers_down_block_idx]
    num_resnets = len(down_block.resnets)
    if down_block.downsamplers is None:
        downsampler = False
    else:
        assert len(down_block.downsamplers) == 1
        downsampler = True
        num_resnets += 1
    for resnet_idx_inc in range(num_resnets):
        full_resnet_prefix = (
            f'{original_down_block_prefix}.{original_down_block_idx + resnet_idx_inc}.0'
            )
        if downsampler and resnet_idx_inc == num_resnets - 1:
            full_diffusers_resnet_prefix = (
                f'down_blocks.{diffusers_down_block_idx}.downsamplers.0')
        else:
            full_diffusers_resnet_prefix = (
                f'{diffusers_resnet_prefix}.{resnet_idx_inc}')
        diffusers_checkpoint.update(resnet_to_diffusers_checkpoint(
            checkpoint, resnet_prefix=full_resnet_prefix,
            diffusers_resnet_prefix=full_diffusers_resnet_prefix))
    if hasattr(down_block, 'attentions'):
        num_attentions = len(down_block.attentions)
        diffusers_attention_prefix = (
            f'down_blocks.{diffusers_down_block_idx}.attentions')
        for attention_idx_inc in range(num_attentions):
            full_attention_prefix = (
                f'{original_down_block_prefix}.{original_down_block_idx + attention_idx_inc}.1'
                )
            full_diffusers_attention_prefix = (
                f'{diffusers_attention_prefix}.{attention_idx_inc}')
            diffusers_checkpoint.update(attention_to_diffusers_checkpoint(
                checkpoint, attention_prefix=full_attention_prefix,
                diffusers_attention_prefix=full_diffusers_attention_prefix,
                num_head_channels=num_head_channels))
    num_original_down_blocks = num_resnets
    return diffusers_checkpoint, num_original_down_blocks

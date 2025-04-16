def movq_encoder_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'encoder.conv_in.weight': checkpoint[
        'encoder.conv_in.weight'], 'encoder.conv_in.bias': checkpoint[
        'encoder.conv_in.bias']})
    for down_block_idx, down_block in enumerate(model.encoder.down_blocks):
        diffusers_down_block_prefix = f'encoder.down_blocks.{down_block_idx}'
        down_block_prefix = f'encoder.down.{down_block_idx}'
        for resnet_idx, resnet in enumerate(down_block.resnets):
            diffusers_resnet_prefix = (
                f'{diffusers_down_block_prefix}.resnets.{resnet_idx}')
            resnet_prefix = f'{down_block_prefix}.block.{resnet_idx}'
            diffusers_checkpoint.update(movq_resnet_to_diffusers_checkpoint
                (resnet, checkpoint, diffusers_resnet_prefix=
                diffusers_resnet_prefix, resnet_prefix=resnet_prefix))
        if down_block_idx != len(model.encoder.down_blocks) - 1:
            diffusers_downsample_prefix = (
                f'{diffusers_down_block_prefix}.downsamplers.0.conv')
            downsample_prefix = f'{down_block_prefix}.downsample.conv'
            diffusers_checkpoint.update({
                f'{diffusers_downsample_prefix}.weight': checkpoint[
                f'{downsample_prefix}.weight'],
                f'{diffusers_downsample_prefix}.bias': checkpoint[
                f'{downsample_prefix}.bias']})
        if hasattr(down_block, 'attentions'):
            for attention_idx, _ in enumerate(down_block.attentions):
                diffusers_attention_prefix = (
                    f'{diffusers_down_block_prefix}.attentions.{attention_idx}'
                    )
                attention_prefix = f'{down_block_prefix}.attn.{attention_idx}'
                diffusers_checkpoint.update(
                    movq_attention_to_diffusers_checkpoint(checkpoint,
                    diffusers_attention_prefix=diffusers_attention_prefix,
                    attention_prefix=attention_prefix))
    diffusers_attention_prefix = 'encoder.mid_block.attentions.0'
    attention_prefix = 'encoder.mid.attn_1'
    diffusers_checkpoint.update(movq_attention_to_diffusers_checkpoint(
        checkpoint, diffusers_attention_prefix=diffusers_attention_prefix,
        attention_prefix=attention_prefix))
    for diffusers_resnet_idx, resnet in enumerate(model.encoder.mid_block.
        resnets):
        diffusers_resnet_prefix = (
            f'encoder.mid_block.resnets.{diffusers_resnet_idx}')
        orig_resnet_idx = diffusers_resnet_idx + 1
        resnet_prefix = f'encoder.mid.block_{orig_resnet_idx}'
        diffusers_checkpoint.update(movq_resnet_to_diffusers_checkpoint(
            resnet, checkpoint, diffusers_resnet_prefix=
            diffusers_resnet_prefix, resnet_prefix=resnet_prefix))
    diffusers_checkpoint.update({'encoder.conv_norm_out.weight': checkpoint
        ['encoder.norm_out.weight'], 'encoder.conv_norm_out.bias':
        checkpoint['encoder.norm_out.bias'], 'encoder.conv_out.weight':
        checkpoint['encoder.conv_out.weight'], 'encoder.conv_out.bias':
        checkpoint['encoder.conv_out.bias']})
    return diffusers_checkpoint

def movq_decoder_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'decoder.conv_in.weight': checkpoint[
        'decoder.conv_in.weight'], 'decoder.conv_in.bias': checkpoint[
        'decoder.conv_in.bias']})
    for diffusers_up_block_idx, up_block in enumerate(model.decoder.up_blocks):
        orig_up_block_idx = len(model.decoder.up_blocks
            ) - 1 - diffusers_up_block_idx
        diffusers_up_block_prefix = (
            f'decoder.up_blocks.{diffusers_up_block_idx}')
        up_block_prefix = f'decoder.up.{orig_up_block_idx}'
        for resnet_idx, resnet in enumerate(up_block.resnets):
            diffusers_resnet_prefix = (
                f'{diffusers_up_block_prefix}.resnets.{resnet_idx}')
            resnet_prefix = f'{up_block_prefix}.block.{resnet_idx}'
            diffusers_checkpoint.update(
                movq_resnet_to_diffusers_checkpoint_spatial_norm(resnet,
                checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix,
                resnet_prefix=resnet_prefix))
        if diffusers_up_block_idx != len(model.decoder.up_blocks) - 1:
            diffusers_downsample_prefix = (
                f'{diffusers_up_block_prefix}.upsamplers.0.conv')
            downsample_prefix = f'{up_block_prefix}.upsample.conv'
            diffusers_checkpoint.update({
                f'{diffusers_downsample_prefix}.weight': checkpoint[
                f'{downsample_prefix}.weight'],
                f'{diffusers_downsample_prefix}.bias': checkpoint[
                f'{downsample_prefix}.bias']})
        if hasattr(up_block, 'attentions'):
            for attention_idx, _ in enumerate(up_block.attentions):
                diffusers_attention_prefix = (
                    f'{diffusers_up_block_prefix}.attentions.{attention_idx}')
                attention_prefix = f'{up_block_prefix}.attn.{attention_idx}'
                diffusers_checkpoint.update(
                    movq_attention_to_diffusers_checkpoint_spatial_norm(
                    checkpoint, diffusers_attention_prefix=
                    diffusers_attention_prefix, attention_prefix=
                    attention_prefix))
    diffusers_attention_prefix = 'decoder.mid_block.attentions.0'
    attention_prefix = 'decoder.mid.attn_1'
    diffusers_checkpoint.update(
        movq_attention_to_diffusers_checkpoint_spatial_norm(checkpoint,
        diffusers_attention_prefix=diffusers_attention_prefix,
        attention_prefix=attention_prefix))
    for diffusers_resnet_idx, resnet in enumerate(model.encoder.mid_block.
        resnets):
        diffusers_resnet_prefix = (
            f'decoder.mid_block.resnets.{diffusers_resnet_idx}')
        orig_resnet_idx = diffusers_resnet_idx + 1
        resnet_prefix = f'decoder.mid.block_{orig_resnet_idx}'
        diffusers_checkpoint.update(
            movq_resnet_to_diffusers_checkpoint_spatial_norm(resnet,
            checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix,
            resnet_prefix=resnet_prefix))
    diffusers_checkpoint.update({'decoder.conv_norm_out.norm_layer.weight':
        checkpoint['decoder.norm_out.norm_layer.weight'],
        'decoder.conv_norm_out.norm_layer.bias': checkpoint[
        'decoder.norm_out.norm_layer.bias'],
        'decoder.conv_norm_out.conv_y.weight': checkpoint[
        'decoder.norm_out.conv_y.weight'],
        'decoder.conv_norm_out.conv_y.bias': checkpoint[
        'decoder.norm_out.conv_y.bias'],
        'decoder.conv_norm_out.conv_b.weight': checkpoint[
        'decoder.norm_out.conv_b.weight'],
        'decoder.conv_norm_out.conv_b.bias': checkpoint[
        'decoder.norm_out.conv_b.bias'], 'decoder.conv_out.weight':
        checkpoint['decoder.conv_out.weight'], 'decoder.conv_out.bias':
        checkpoint['decoder.conv_out.bias']})
    return diffusers_checkpoint

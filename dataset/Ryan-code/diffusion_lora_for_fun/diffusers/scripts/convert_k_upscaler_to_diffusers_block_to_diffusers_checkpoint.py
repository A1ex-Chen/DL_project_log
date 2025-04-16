def block_to_diffusers_checkpoint(block, checkpoint, block_idx, block_type):
    block_prefix = ('inner_model.u_net.u_blocks' if block_type == 'up' else
        'inner_model.u_net.d_blocks')
    block_prefix = f'{block_prefix}.{block_idx}'
    diffusers_checkpoint = {}
    if not hasattr(block, 'attentions'):
        n = 1
    elif not block.attentions[0].add_self_attention:
        n = 2
    else:
        n = 3
    for resnet_idx, resnet in enumerate(block.resnets):
        diffusers_resnet_prefix = (
            f'{block_type}_blocks.{block_idx}.resnets.{resnet_idx}')
        idx = n * resnet_idx if block_type == 'up' else n * resnet_idx + 1
        resnet_prefix = (f'{block_prefix}.{idx}' if block_type == 'up' else
            f'{block_prefix}.{idx}')
        diffusers_checkpoint.update(resnet_to_diffusers_checkpoint(resnet,
            checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix,
            resnet_prefix=resnet_prefix))
    if hasattr(block, 'attentions'):
        for attention_idx, attention in enumerate(block.attentions):
            diffusers_attention_prefix = (
                f'{block_type}_blocks.{block_idx}.attentions.{attention_idx}')
            idx = (n * attention_idx + 1 if block_type == 'up' else n *
                attention_idx + 2)
            self_attention_prefix = f'{block_prefix}.{idx}'
            cross_attention_prefix = f'{block_prefix}.{idx}'
            cross_attention_index = (1 if not attention.add_self_attention else
                2)
            idx = (n * attention_idx + cross_attention_index if block_type ==
                'up' else n * attention_idx + cross_attention_index + 1)
            cross_attention_prefix = f'{block_prefix}.{idx}'
            diffusers_checkpoint.update(cross_attn_to_diffusers_checkpoint(
                checkpoint, diffusers_attention_prefix=
                diffusers_attention_prefix, diffusers_attention_index=2,
                attention_prefix=cross_attention_prefix))
            if attention.add_self_attention is True:
                diffusers_checkpoint.update(self_attn_to_diffusers_checkpoint
                    (checkpoint, diffusers_attention_prefix=
                    diffusers_attention_prefix, attention_prefix=
                    self_attention_prefix))
    return diffusers_checkpoint

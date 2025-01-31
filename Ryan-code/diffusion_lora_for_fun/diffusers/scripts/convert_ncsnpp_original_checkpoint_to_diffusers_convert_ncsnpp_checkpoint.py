def convert_ncsnpp_checkpoint(checkpoint, config):
    """
    Takes a state dict and the path to
    """
    new_model_architecture = UNet2DModel(**config)
    new_model_architecture.time_proj.W.data = checkpoint['all_modules.0.W'
        ].data
    new_model_architecture.time_proj.weight.data = checkpoint['all_modules.0.W'
        ].data
    new_model_architecture.time_embedding.linear_1.weight.data = checkpoint[
        'all_modules.1.weight'].data
    new_model_architecture.time_embedding.linear_1.bias.data = checkpoint[
        'all_modules.1.bias'].data
    new_model_architecture.time_embedding.linear_2.weight.data = checkpoint[
        'all_modules.2.weight'].data
    new_model_architecture.time_embedding.linear_2.bias.data = checkpoint[
        'all_modules.2.bias'].data
    new_model_architecture.conv_in.weight.data = checkpoint[
        'all_modules.3.weight'].data
    new_model_architecture.conv_in.bias.data = checkpoint['all_modules.3.bias'
        ].data
    new_model_architecture.conv_norm_out.weight.data = checkpoint[list(
        checkpoint.keys())[-4]].data
    new_model_architecture.conv_norm_out.bias.data = checkpoint[list(
        checkpoint.keys())[-3]].data
    new_model_architecture.conv_out.weight.data = checkpoint[list(
        checkpoint.keys())[-2]].data
    new_model_architecture.conv_out.bias.data = checkpoint[list(checkpoint.
        keys())[-1]].data
    module_index = 4

    def set_attention_weights(new_layer, old_checkpoint, index):
        new_layer.query.weight.data = old_checkpoint[
            f'all_modules.{index}.NIN_0.W'].data.T
        new_layer.key.weight.data = old_checkpoint[
            f'all_modules.{index}.NIN_1.W'].data.T
        new_layer.value.weight.data = old_checkpoint[
            f'all_modules.{index}.NIN_2.W'].data.T
        new_layer.query.bias.data = old_checkpoint[
            f'all_modules.{index}.NIN_0.b'].data
        new_layer.key.bias.data = old_checkpoint[f'all_modules.{index}.NIN_1.b'
            ].data
        new_layer.value.bias.data = old_checkpoint[
            f'all_modules.{index}.NIN_2.b'].data
        new_layer.proj_attn.weight.data = old_checkpoint[
            f'all_modules.{index}.NIN_3.W'].data.T
        new_layer.proj_attn.bias.data = old_checkpoint[
            f'all_modules.{index}.NIN_3.b'].data
        new_layer.group_norm.weight.data = old_checkpoint[
            f'all_modules.{index}.GroupNorm_0.weight'].data
        new_layer.group_norm.bias.data = old_checkpoint[
            f'all_modules.{index}.GroupNorm_0.bias'].data

    def set_resnet_weights(new_layer, old_checkpoint, index):
        new_layer.conv1.weight.data = old_checkpoint[
            f'all_modules.{index}.Conv_0.weight'].data
        new_layer.conv1.bias.data = old_checkpoint[
            f'all_modules.{index}.Conv_0.bias'].data
        new_layer.norm1.weight.data = old_checkpoint[
            f'all_modules.{index}.GroupNorm_0.weight'].data
        new_layer.norm1.bias.data = old_checkpoint[
            f'all_modules.{index}.GroupNorm_0.bias'].data
        new_layer.conv2.weight.data = old_checkpoint[
            f'all_modules.{index}.Conv_1.weight'].data
        new_layer.conv2.bias.data = old_checkpoint[
            f'all_modules.{index}.Conv_1.bias'].data
        new_layer.norm2.weight.data = old_checkpoint[
            f'all_modules.{index}.GroupNorm_1.weight'].data
        new_layer.norm2.bias.data = old_checkpoint[
            f'all_modules.{index}.GroupNorm_1.bias'].data
        new_layer.time_emb_proj.weight.data = old_checkpoint[
            f'all_modules.{index}.Dense_0.weight'].data
        new_layer.time_emb_proj.bias.data = old_checkpoint[
            f'all_modules.{index}.Dense_0.bias'].data
        if (new_layer.in_channels != new_layer.out_channels or new_layer.up or
            new_layer.down):
            new_layer.conv_shortcut.weight.data = old_checkpoint[
                f'all_modules.{index}.Conv_2.weight'].data
            new_layer.conv_shortcut.bias.data = old_checkpoint[
                f'all_modules.{index}.Conv_2.bias'].data
    for i, block in enumerate(new_model_architecture.downsample_blocks):
        has_attentions = hasattr(block, 'attentions')
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j], checkpoint, module_index)
            module_index += 1
            if has_attentions:
                set_attention_weights(block.attentions[j], checkpoint,
                    module_index)
                module_index += 1
        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            set_resnet_weights(block.resnet_down, checkpoint, module_index)
            module_index += 1
            block.skip_conv.weight.data = checkpoint[
                f'all_modules.{module_index}.Conv_0.weight'].data
            block.skip_conv.bias.data = checkpoint[
                f'all_modules.{module_index}.Conv_0.bias'].data
            module_index += 1
    set_resnet_weights(new_model_architecture.mid_block.resnets[0],
        checkpoint, module_index)
    module_index += 1
    set_attention_weights(new_model_architecture.mid_block.attentions[0],
        checkpoint, module_index)
    module_index += 1
    set_resnet_weights(new_model_architecture.mid_block.resnets[1],
        checkpoint, module_index)
    module_index += 1
    for i, block in enumerate(new_model_architecture.up_blocks):
        has_attentions = hasattr(block, 'attentions')
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j], checkpoint, module_index)
            module_index += 1
        if has_attentions:
            set_attention_weights(block.attentions[0], checkpoint, module_index
                )
            module_index += 1
        if hasattr(block, 'resnet_up') and block.resnet_up is not None:
            block.skip_norm.weight.data = checkpoint[
                f'all_modules.{module_index}.weight'].data
            block.skip_norm.bias.data = checkpoint[
                f'all_modules.{module_index}.bias'].data
            module_index += 1
            block.skip_conv.weight.data = checkpoint[
                f'all_modules.{module_index}.weight'].data
            block.skip_conv.bias.data = checkpoint[
                f'all_modules.{module_index}.bias'].data
            module_index += 1
            set_resnet_weights(block.resnet_up, checkpoint, module_index)
            module_index += 1
    new_model_architecture.conv_norm_out.weight.data = checkpoint[
        f'all_modules.{module_index}.weight'].data
    new_model_architecture.conv_norm_out.bias.data = checkpoint[
        f'all_modules.{module_index}.bias'].data
    module_index += 1
    new_model_architecture.conv_out.weight.data = checkpoint[
        f'all_modules.{module_index}.weight'].data
    new_model_architecture.conv_out.bias.data = checkpoint[
        f'all_modules.{module_index}.bias'].data
    return new_model_architecture.state_dict()

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

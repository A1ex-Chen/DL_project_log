def movq_resnet_to_diffusers_checkpoint_spatial_norm(resnet, checkpoint, *,
    diffusers_resnet_prefix, resnet_prefix):
    rv = {f'{diffusers_resnet_prefix}.norm1.norm_layer.weight': checkpoint[
        f'{resnet_prefix}.norm1.norm_layer.weight'],
        f'{diffusers_resnet_prefix}.norm1.norm_layer.bias': checkpoint[
        f'{resnet_prefix}.norm1.norm_layer.bias'],
        f'{diffusers_resnet_prefix}.norm1.conv_y.weight': checkpoint[
        f'{resnet_prefix}.norm1.conv_y.weight'],
        f'{diffusers_resnet_prefix}.norm1.conv_y.bias': checkpoint[
        f'{resnet_prefix}.norm1.conv_y.bias'],
        f'{diffusers_resnet_prefix}.norm1.conv_b.weight': checkpoint[
        f'{resnet_prefix}.norm1.conv_b.weight'],
        f'{diffusers_resnet_prefix}.norm1.conv_b.bias': checkpoint[
        f'{resnet_prefix}.norm1.conv_b.bias'],
        f'{diffusers_resnet_prefix}.conv1.weight': checkpoint[
        f'{resnet_prefix}.conv1.weight'],
        f'{diffusers_resnet_prefix}.conv1.bias': checkpoint[
        f'{resnet_prefix}.conv1.bias'],
        f'{diffusers_resnet_prefix}.norm2.norm_layer.weight': checkpoint[
        f'{resnet_prefix}.norm2.norm_layer.weight'],
        f'{diffusers_resnet_prefix}.norm2.norm_layer.bias': checkpoint[
        f'{resnet_prefix}.norm2.norm_layer.bias'],
        f'{diffusers_resnet_prefix}.norm2.conv_y.weight': checkpoint[
        f'{resnet_prefix}.norm2.conv_y.weight'],
        f'{diffusers_resnet_prefix}.norm2.conv_y.bias': checkpoint[
        f'{resnet_prefix}.norm2.conv_y.bias'],
        f'{diffusers_resnet_prefix}.norm2.conv_b.weight': checkpoint[
        f'{resnet_prefix}.norm2.conv_b.weight'],
        f'{diffusers_resnet_prefix}.norm2.conv_b.bias': checkpoint[
        f'{resnet_prefix}.norm2.conv_b.bias'],
        f'{diffusers_resnet_prefix}.conv2.weight': checkpoint[
        f'{resnet_prefix}.conv2.weight'],
        f'{diffusers_resnet_prefix}.conv2.bias': checkpoint[
        f'{resnet_prefix}.conv2.bias']}
    if resnet.conv_shortcut is not None:
        rv.update({f'{diffusers_resnet_prefix}.conv_shortcut.weight':
            checkpoint[f'{resnet_prefix}.nin_shortcut.weight'],
            f'{diffusers_resnet_prefix}.conv_shortcut.bias': checkpoint[
            f'{resnet_prefix}.nin_shortcut.bias']})
    return rv

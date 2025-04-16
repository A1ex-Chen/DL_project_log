def resnet_to_diffusers_checkpoint(resnet, checkpoint, *,
    diffusers_resnet_prefix, resnet_prefix):
    rv = {f'{diffusers_resnet_prefix}.norm1.linear.weight': checkpoint[
        f'{resnet_prefix}.main.0.mapper.weight'],
        f'{diffusers_resnet_prefix}.norm1.linear.bias': checkpoint[
        f'{resnet_prefix}.main.0.mapper.bias'],
        f'{diffusers_resnet_prefix}.conv1.weight': checkpoint[
        f'{resnet_prefix}.main.2.weight'],
        f'{diffusers_resnet_prefix}.conv1.bias': checkpoint[
        f'{resnet_prefix}.main.2.bias'],
        f'{diffusers_resnet_prefix}.norm2.linear.weight': checkpoint[
        f'{resnet_prefix}.main.4.mapper.weight'],
        f'{diffusers_resnet_prefix}.norm2.linear.bias': checkpoint[
        f'{resnet_prefix}.main.4.mapper.bias'],
        f'{diffusers_resnet_prefix}.conv2.weight': checkpoint[
        f'{resnet_prefix}.main.6.weight'],
        f'{diffusers_resnet_prefix}.conv2.bias': checkpoint[
        f'{resnet_prefix}.main.6.bias']}
    if resnet.conv_shortcut is not None:
        rv.update({f'{diffusers_resnet_prefix}.conv_shortcut.weight':
            checkpoint[f'{resnet_prefix}.skip.weight']})
    return rv

def movq_resnet_to_diffusers_checkpoint(resnet, checkpoint, *,
    diffusers_resnet_prefix, resnet_prefix):
    rv = {f'{diffusers_resnet_prefix}.norm1.weight': checkpoint[
        f'{resnet_prefix}.norm1.weight'],
        f'{diffusers_resnet_prefix}.norm1.bias': checkpoint[
        f'{resnet_prefix}.norm1.bias'],
        f'{diffusers_resnet_prefix}.conv1.weight': checkpoint[
        f'{resnet_prefix}.conv1.weight'],
        f'{diffusers_resnet_prefix}.conv1.bias': checkpoint[
        f'{resnet_prefix}.conv1.bias'],
        f'{diffusers_resnet_prefix}.norm2.weight': checkpoint[
        f'{resnet_prefix}.norm2.weight'],
        f'{diffusers_resnet_prefix}.norm2.bias': checkpoint[
        f'{resnet_prefix}.norm2.bias'],
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

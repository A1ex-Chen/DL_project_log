def resnet_to_diffusers_checkpoint(checkpoint, *, diffusers_resnet_prefix,
    resnet_prefix):
    diffusers_checkpoint = {f'{diffusers_resnet_prefix}.norm1.weight':
        checkpoint[f'{resnet_prefix}.in_layers.0.weight'],
        f'{diffusers_resnet_prefix}.norm1.bias': checkpoint[
        f'{resnet_prefix}.in_layers.0.bias'],
        f'{diffusers_resnet_prefix}.conv1.weight': checkpoint[
        f'{resnet_prefix}.in_layers.2.weight'],
        f'{diffusers_resnet_prefix}.conv1.bias': checkpoint[
        f'{resnet_prefix}.in_layers.2.bias'],
        f'{diffusers_resnet_prefix}.time_emb_proj.weight': checkpoint[
        f'{resnet_prefix}.emb_layers.1.weight'],
        f'{diffusers_resnet_prefix}.time_emb_proj.bias': checkpoint[
        f'{resnet_prefix}.emb_layers.1.bias'],
        f'{diffusers_resnet_prefix}.norm2.weight': checkpoint[
        f'{resnet_prefix}.out_layers.0.weight'],
        f'{diffusers_resnet_prefix}.norm2.bias': checkpoint[
        f'{resnet_prefix}.out_layers.0.bias'],
        f'{diffusers_resnet_prefix}.conv2.weight': checkpoint[
        f'{resnet_prefix}.out_layers.3.weight'],
        f'{diffusers_resnet_prefix}.conv2.bias': checkpoint[
        f'{resnet_prefix}.out_layers.3.bias']}
    skip_connection_prefix = f'{resnet_prefix}.skip_connection'
    if f'{skip_connection_prefix}.weight' in checkpoint:
        diffusers_checkpoint.update({
            f'{diffusers_resnet_prefix}.conv_shortcut.weight': checkpoint[
            f'{skip_connection_prefix}.weight'],
            f'{diffusers_resnet_prefix}.conv_shortcut.bias': checkpoint[
            f'{skip_connection_prefix}.bias']})
    return diffusers_checkpoint

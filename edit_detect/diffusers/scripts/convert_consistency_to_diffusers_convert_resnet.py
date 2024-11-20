def convert_resnet(checkpoint, new_checkpoint, old_prefix, new_prefix,
    has_skip=False):
    new_checkpoint[f'{new_prefix}.norm1.weight'] = checkpoint[
        f'{old_prefix}.in_layers.0.weight']
    new_checkpoint[f'{new_prefix}.norm1.bias'] = checkpoint[
        f'{old_prefix}.in_layers.0.bias']
    new_checkpoint[f'{new_prefix}.conv1.weight'] = checkpoint[
        f'{old_prefix}.in_layers.2.weight']
    new_checkpoint[f'{new_prefix}.conv1.bias'] = checkpoint[
        f'{old_prefix}.in_layers.2.bias']
    new_checkpoint[f'{new_prefix}.time_emb_proj.weight'] = checkpoint[
        f'{old_prefix}.emb_layers.1.weight']
    new_checkpoint[f'{new_prefix}.time_emb_proj.bias'] = checkpoint[
        f'{old_prefix}.emb_layers.1.bias']
    new_checkpoint[f'{new_prefix}.norm2.weight'] = checkpoint[
        f'{old_prefix}.out_layers.0.weight']
    new_checkpoint[f'{new_prefix}.norm2.bias'] = checkpoint[
        f'{old_prefix}.out_layers.0.bias']
    new_checkpoint[f'{new_prefix}.conv2.weight'] = checkpoint[
        f'{old_prefix}.out_layers.3.weight']
    new_checkpoint[f'{new_prefix}.conv2.bias'] = checkpoint[
        f'{old_prefix}.out_layers.3.bias']
    if has_skip:
        new_checkpoint[f'{new_prefix}.conv_shortcut.weight'] = checkpoint[
            f'{old_prefix}.skip_connection.weight']
        new_checkpoint[f'{new_prefix}.conv_shortcut.bias'] = checkpoint[
            f'{old_prefix}.skip_connection.bias']
    return new_checkpoint

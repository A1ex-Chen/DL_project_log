def convert_attention(checkpoint, new_checkpoint, old_prefix, new_prefix,
    attention_dim=None):
    weight_q, weight_k, weight_v = checkpoint[f'{old_prefix}.qkv.weight'
        ].chunk(3, dim=0)
    bias_q, bias_k, bias_v = checkpoint[f'{old_prefix}.qkv.bias'].chunk(3,
        dim=0)
    new_checkpoint[f'{new_prefix}.group_norm.weight'] = checkpoint[
        f'{old_prefix}.norm.weight']
    new_checkpoint[f'{new_prefix}.group_norm.bias'] = checkpoint[
        f'{old_prefix}.norm.bias']
    new_checkpoint[f'{new_prefix}.to_q.weight'] = weight_q.squeeze(-1).squeeze(
        -1)
    new_checkpoint[f'{new_prefix}.to_q.bias'] = bias_q.squeeze(-1).squeeze(-1)
    new_checkpoint[f'{new_prefix}.to_k.weight'] = weight_k.squeeze(-1).squeeze(
        -1)
    new_checkpoint[f'{new_prefix}.to_k.bias'] = bias_k.squeeze(-1).squeeze(-1)
    new_checkpoint[f'{new_prefix}.to_v.weight'] = weight_v.squeeze(-1).squeeze(
        -1)
    new_checkpoint[f'{new_prefix}.to_v.bias'] = bias_v.squeeze(-1).squeeze(-1)
    new_checkpoint[f'{new_prefix}.to_out.0.weight'] = checkpoint[
        f'{old_prefix}.proj_out.weight'].squeeze(-1).squeeze(-1)
    new_checkpoint[f'{new_prefix}.to_out.0.bias'] = checkpoint[
        f'{old_prefix}.proj_out.bias'].squeeze(-1).squeeze(-1)
    return new_checkpoint

def convert_uvit_block_to_diffusers_block(uvit_state_dict, new_state_dict,
    block_prefix, new_prefix='transformer.transformer_', skip_connection=False
    ):
    """
    Maps the keys in a UniDiffuser transformer block (`Block`) to the keys in a diffusers transformer block
    (`UTransformerBlock`/`UniDiffuserBlock`).
    """
    prefix = new_prefix + block_prefix
    if skip_connection:
        new_state_dict[prefix + '.skip.skip_linear.weight'] = uvit_state_dict[
            block_prefix + '.skip_linear.weight']
        new_state_dict[prefix + '.skip.skip_linear.bias'] = uvit_state_dict[
            block_prefix + '.skip_linear.bias']
        new_state_dict[prefix + '.skip.norm.weight'] = uvit_state_dict[
            block_prefix + '.norm1.weight']
        new_state_dict[prefix + '.skip.norm.bias'] = uvit_state_dict[
            block_prefix + '.norm1.bias']
        prefix += '.block'
    qkv = uvit_state_dict[block_prefix + '.attn.qkv.weight']
    new_attn_keys = ['.attn1.to_q.weight', '.attn1.to_k.weight',
        '.attn1.to_v.weight']
    new_attn_keys = [(prefix + key) for key in new_attn_keys]
    shape = qkv.shape[0] // len(new_attn_keys)
    for i, attn_key in enumerate(new_attn_keys):
        new_state_dict[attn_key] = qkv[i * shape:(i + 1) * shape]
    new_state_dict[prefix + '.attn1.to_out.0.weight'] = uvit_state_dict[
        block_prefix + '.attn.proj.weight']
    new_state_dict[prefix + '.attn1.to_out.0.bias'] = uvit_state_dict[
        block_prefix + '.attn.proj.bias']
    new_state_dict[prefix + '.norm1.weight'] = uvit_state_dict[block_prefix +
        '.norm2.weight']
    new_state_dict[prefix + '.norm1.bias'] = uvit_state_dict[block_prefix +
        '.norm2.bias']
    new_state_dict[prefix + '.ff.net.0.proj.weight'] = uvit_state_dict[
        block_prefix + '.mlp.fc1.weight']
    new_state_dict[prefix + '.ff.net.0.proj.bias'] = uvit_state_dict[
        block_prefix + '.mlp.fc1.bias']
    new_state_dict[prefix + '.ff.net.2.weight'] = uvit_state_dict[
        block_prefix + '.mlp.fc2.weight']
    new_state_dict[prefix + '.ff.net.2.bias'] = uvit_state_dict[
        block_prefix + '.mlp.fc2.bias']
    new_state_dict[prefix + '.norm3.weight'] = uvit_state_dict[block_prefix +
        '.norm3.weight']
    new_state_dict[prefix + '.norm3.bias'] = uvit_state_dict[block_prefix +
        '.norm3.bias']
    return uvit_state_dict, new_state_dict

def convert_uvit_to_diffusers(ckpt, diffusers_model):
    """
    Converts a UniDiffuser uvit_v*.pth checkpoint to a diffusers UniDiffusersModel.
    """
    uvit_state_dict = torch.load(ckpt, map_location='cpu')
    new_state_dict = {}
    new_state_dict['vae_img_in.proj.weight'] = uvit_state_dict[
        'patch_embed.proj.weight']
    new_state_dict['vae_img_in.proj.bias'] = uvit_state_dict[
        'patch_embed.proj.bias']
    new_state_dict['clip_img_in.weight'] = uvit_state_dict[
        'clip_img_embed.weight']
    new_state_dict['clip_img_in.bias'] = uvit_state_dict['clip_img_embed.bias']
    new_state_dict['text_in.weight'] = uvit_state_dict['text_embed.weight']
    new_state_dict['text_in.bias'] = uvit_state_dict['text_embed.bias']
    new_state_dict['pos_embed'] = uvit_state_dict['pos_embed']
    if ('token_embedding.weight' in uvit_state_dict and diffusers_model.
        use_data_type_embedding):
        new_state_dict['data_type_pos_embed_token'] = uvit_state_dict[
            'pos_embed_token']
        new_state_dict['data_type_token_embedding.weight'] = uvit_state_dict[
            'token_embedding.weight']
    new_state_dict['transformer.pos_embed.proj.weight'] = uvit_state_dict[
        'patch_embed.proj.weight']
    new_state_dict['transformer.pos_embed.proj.bias'] = uvit_state_dict[
        'patch_embed.proj.bias']
    new_state_dict['transformer.norm_out.weight'] = uvit_state_dict[
        'norm.weight']
    new_state_dict['transformer.norm_out.bias'] = uvit_state_dict['norm.bias']
    new_state_dict['vae_img_out.weight'] = uvit_state_dict[
        'decoder_pred.weight']
    new_state_dict['vae_img_out.bias'] = uvit_state_dict['decoder_pred.bias']
    new_state_dict['clip_img_out.weight'] = uvit_state_dict[
        'clip_img_out.weight']
    new_state_dict['clip_img_out.bias'] = uvit_state_dict['clip_img_out.bias']
    new_state_dict['text_out.weight'] = uvit_state_dict['text_out.weight']
    new_state_dict['text_out.bias'] = uvit_state_dict['text_out.bias']
    in_blocks_prefixes = {'.'.join(layer.split('.')[:2]) for layer in
        uvit_state_dict if 'in_blocks' in layer}
    for in_block_prefix in list(in_blocks_prefixes):
        convert_uvit_block_to_diffusers_block(uvit_state_dict,
            new_state_dict, in_block_prefix)
    convert_uvit_block_to_diffusers_block(uvit_state_dict, new_state_dict,
        'mid_block')
    out_blocks_prefixes = {'.'.join(layer.split('.')[:2]) for layer in
        uvit_state_dict if 'out_blocks' in layer}
    for out_block_prefix in list(out_blocks_prefixes):
        convert_uvit_block_to_diffusers_block(uvit_state_dict,
            new_state_dict, out_block_prefix, skip_connection=True)
    missing_keys, unexpected_keys = diffusers_model.load_state_dict(
        new_state_dict)
    for missing_key in missing_keys:
        print(f'Missing key: {missing_key}')
    for unexpected_key in unexpected_keys:
        print(f'Unexpected key: {unexpected_key}')
    return diffusers_model

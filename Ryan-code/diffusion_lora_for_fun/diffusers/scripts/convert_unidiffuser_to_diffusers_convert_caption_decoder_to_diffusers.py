def convert_caption_decoder_to_diffusers(ckpt, diffusers_model):
    """
    Converts a UniDiffuser caption_decoder.pth checkpoint to a diffusers UniDiffuserTextDecoder.
    """
    checkpoint_state_dict = torch.load(ckpt, map_location='cpu')
    decoder_state_dict = {}
    caption_decoder_key = 'module.'
    for key in checkpoint_state_dict:
        if key.startswith(caption_decoder_key):
            decoder_state_dict[key.replace(caption_decoder_key, '')
                ] = checkpoint_state_dict.get(key)
        else:
            decoder_state_dict[key] = checkpoint_state_dict.get(key)
    new_state_dict = {}
    new_state_dict['encode_prefix.weight'] = decoder_state_dict[
        'encode_prefix.weight']
    new_state_dict['encode_prefix.bias'] = decoder_state_dict[
        'encode_prefix.bias']
    new_state_dict['decode_prefix.weight'] = decoder_state_dict[
        'decode_prefix.weight']
    new_state_dict['decode_prefix.bias'] = decoder_state_dict[
        'decode_prefix.bias']
    for key, val in decoder_state_dict.items():
        if key.startswith('gpt'):
            suffix = key[len('gpt'):]
            new_state_dict['transformer' + suffix] = val
    missing_keys, unexpected_keys = diffusers_model.load_state_dict(
        new_state_dict)
    for missing_key in missing_keys:
        print(f'Missing key: {missing_key}')
    for unexpected_key in unexpected_keys:
        print(f'Unexpected key: {unexpected_key}')
    return diffusers_model

def load_from_pretrained(self, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=
            False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict
        ['visual_encoder.pos_embed'], self.visual_encoder)
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self', 'self0')
            new_key1 = key.replace('self', 'self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense', 'dense0')
            new_key1 = key.replace('dense', 'dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
    msg = self.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    print(f'missing keys {msg.missing_keys}')
    return msg

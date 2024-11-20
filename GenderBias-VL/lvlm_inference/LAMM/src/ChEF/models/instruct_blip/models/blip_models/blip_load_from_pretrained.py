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
    if 'visual_encoder_m.pos_embed' in self.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], self.visual_encoder_m)
    for key in self.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != self.state_dict()[key].shape:
                del state_dict[key]
    msg = self.load_state_dict(state_dict, strict=False)
    logging.info('Missing keys {}'.format(msg.missing_keys))
    logging.info('load checkpoint from %s' % url_or_filename)
    return msg

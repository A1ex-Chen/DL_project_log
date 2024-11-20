def load_checkpoint(self, url_or_filename):
    """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=
            False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    msg = self.load_state_dict(state_dict, strict=False)
    logging.info('Missing keys {}'.format(msg.missing_keys))
    logging.info('load checkpoint from %s' % url_or_filename)
    return msg

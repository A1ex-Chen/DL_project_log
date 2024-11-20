def rename_old_checkpoint(old_ckpt: dict[str, torch.Tensor]) ->dict[str,
    torch.Tensor]:
    """Rename some keys in the old checkpoint"""
    perceiver_pattern1 = re.compile('perceiver\\.layers\\.[0-9]\\.0')
    perceiver_pattern2 = re.compile('perceiver\\.layers\\.[0-9]\\.1')
    new_ckpt = old_ckpt.copy()
    for key, value in old_ckpt.items():
        if re.match(perceiver_pattern1, key):
            new_key = re.sub('([0-9])\\.0', '\\1', key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif re.match(perceiver_pattern2, key):
            new_key = re.sub('([0-9])\\.1', '\\1.feed_forward', key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif key.startswith('lang_encoder.gated_cross_attn_layers.'):
            new_ckpt.pop(key)
        elif key.startswith('lang_encoder.') and 'ff_gate' not in key:
            new_key = key.replace('ff', 'feed_forward')
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
    return new_ckpt

def is_open_clip_model(checkpoint):
    if CHECKPOINT_KEY_NAMES['open_clip'] in checkpoint:
        return True
    return False

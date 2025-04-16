def is_clip_model(checkpoint):
    if CHECKPOINT_KEY_NAMES['clip'] in checkpoint:
        return True
    return False

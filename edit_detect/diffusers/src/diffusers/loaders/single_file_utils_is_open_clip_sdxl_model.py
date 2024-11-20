def is_open_clip_sdxl_model(checkpoint):
    if CHECKPOINT_KEY_NAMES['open_clip_sdxl'] in checkpoint:
        return True
    return False

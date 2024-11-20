def is_clip_sdxl_model(checkpoint):
    if CHECKPOINT_KEY_NAMES['clip_sdxl'] in checkpoint:
        return True
    return False

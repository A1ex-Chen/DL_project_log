def is_open_clip_sdxl_refiner_model(checkpoint):
    if CHECKPOINT_KEY_NAMES['open_clip_sdxl_refiner'] in checkpoint:
        return True
    return False

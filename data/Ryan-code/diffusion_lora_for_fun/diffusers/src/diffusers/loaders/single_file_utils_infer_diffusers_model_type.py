def infer_diffusers_model_type(checkpoint):
    if CHECKPOINT_KEY_NAMES['inpainting'] in checkpoint and checkpoint[
        CHECKPOINT_KEY_NAMES['inpainting']].shape[1] == 9:
        if CHECKPOINT_KEY_NAMES['v2'] in checkpoint and checkpoint[
            CHECKPOINT_KEY_NAMES['v2']].shape[-1] == 1024:
            model_type = 'inpainting_v2'
        else:
            model_type = 'inpainting'
    elif CHECKPOINT_KEY_NAMES['v2'] in checkpoint and checkpoint[
        CHECKPOINT_KEY_NAMES['v2']].shape[-1] == 1024:
        model_type = 'v2'
    elif CHECKPOINT_KEY_NAMES['playground-v2-5'] in checkpoint:
        model_type = 'playground-v2-5'
    elif CHECKPOINT_KEY_NAMES['xl_base'] in checkpoint:
        model_type = 'xl_base'
    elif CHECKPOINT_KEY_NAMES['xl_refiner'] in checkpoint:
        model_type = 'xl_refiner'
    elif CHECKPOINT_KEY_NAMES['upscale'] in checkpoint:
        model_type = 'upscale'
    elif CHECKPOINT_KEY_NAMES['controlnet'] in checkpoint:
        model_type = 'controlnet'
    elif CHECKPOINT_KEY_NAMES['stable_cascade_stage_c'
        ] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES[
        'stable_cascade_stage_c']].shape[0] == 1536:
        model_type = 'stable_cascade_stage_c_lite'
    elif CHECKPOINT_KEY_NAMES['stable_cascade_stage_c'
        ] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES[
        'stable_cascade_stage_c']].shape[0] == 2048:
        model_type = 'stable_cascade_stage_c'
    elif CHECKPOINT_KEY_NAMES['stable_cascade_stage_b'
        ] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES[
        'stable_cascade_stage_b']].shape[-1] == 576:
        model_type = 'stable_cascade_stage_b_lite'
    elif CHECKPOINT_KEY_NAMES['stable_cascade_stage_b'
        ] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES[
        'stable_cascade_stage_b']].shape[-1] == 640:
        model_type = 'stable_cascade_stage_b'
    else:
        model_type = 'v1'
    return model_type

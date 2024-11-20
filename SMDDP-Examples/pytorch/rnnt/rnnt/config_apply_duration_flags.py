def apply_duration_flags(cfg, max_duration):
    if max_duration is not None:
        cfg['input_train']['audio_dataset']['max_duration'] = max_duration

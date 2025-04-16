def create_pann_model(audio_cfg, enable_fusion=False, fusion_type='None'):
    try:
        ModelProto = eval(audio_cfg.model_name)
        model = ModelProto(sample_rate=audio_cfg.sample_rate, window_size=
            audio_cfg.window_size, hop_size=audio_cfg.hop_size, mel_bins=
            audio_cfg.mel_bins, fmin=audio_cfg.fmin, fmax=audio_cfg.fmax,
            classes_num=audio_cfg.class_num, enable_fusion=enable_fusion,
            fusion_type=fusion_type)
        return model
    except:
        raise RuntimeError(
            f'Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough.'
            )

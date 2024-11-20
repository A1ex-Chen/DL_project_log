def create_htsat_model(audio_cfg, enable_fusion=False, fusion_type='None'):
    try:
        assert audio_cfg.model_name in ['tiny', 'base', 'large'
            ], 'model name for HTS-AT is wrong!'
        if audio_cfg.model_name == 'tiny':
            model = HTSAT_Swin_Transformer(spec_size=256, patch_size=4,
                patch_stride=(4, 4), num_classes=audio_cfg.class_num,
                embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                window_size=8, config=audio_cfg, enable_fusion=
                enable_fusion, fusion_type=fusion_type)
        elif audio_cfg.model_name == 'base':
            model = HTSAT_Swin_Transformer(spec_size=256, patch_size=4,
                patch_stride=(4, 4), num_classes=audio_cfg.class_num,
                embed_dim=128, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 
                32], window_size=8, config=audio_cfg, enable_fusion=
                enable_fusion, fusion_type=fusion_type)
        elif audio_cfg.model_name == 'large':
            model = HTSAT_Swin_Transformer(spec_size=256, patch_size=4,
                patch_stride=(4, 4), num_classes=audio_cfg.class_num,
                embed_dim=256, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 
                32], window_size=8, config=audio_cfg, enable_fusion=
                enable_fusion, fusion_type=fusion_type)
        return model
    except:
        raise RuntimeError(
            f'Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough.'
            )

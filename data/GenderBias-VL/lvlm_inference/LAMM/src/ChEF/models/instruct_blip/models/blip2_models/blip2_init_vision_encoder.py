def init_vision_encoder(self, model_name, img_size, drop_path_rate,
    use_grad_checkpoint, precision):
    assert model_name in ['eva_clip_g', 'eva2_clip_L', 'clip_L'
        ], 'vit model must be eva_clip_g, eva2_clip_L or clip_L'
    if model_name == 'eva_clip_g':
        visual_encoder = create_eva_vit_g(img_size, drop_path_rate,
            use_grad_checkpoint, precision)
    elif model_name == 'clip_L':
        visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint,
            precision)
    ln_vision = LayerNorm(visual_encoder.num_features)
    self.vit_name = model_name
    return visual_encoder, ln_vision

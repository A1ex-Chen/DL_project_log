@classmethod
def init_vision_encoder(cls, model_name, img_size, drop_path_rate,
    use_grad_checkpoint, precision):
    assert model_name == 'eva_clip_g', 'vit model must be eva_clip_g for current version of MiniGPT-4'
    visual_encoder = create_eva_vit_g(img_size, drop_path_rate,
        use_grad_checkpoint, precision)
    ln_vision = LayerNorm(visual_encoder.num_features)
    return visual_encoder, ln_vision

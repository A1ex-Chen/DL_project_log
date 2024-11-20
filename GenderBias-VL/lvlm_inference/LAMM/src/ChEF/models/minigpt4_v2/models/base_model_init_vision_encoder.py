@classmethod
def init_vision_encoder(cls, model_name, img_size, drop_path_rate,
    use_grad_checkpoint, precision, freeze):
    logging.info('Loading VIT')
    assert model_name == 'eva_clip_g', 'vit model must be eva_clip_g for current version of MiniGPT-4'
    if not freeze:
        precision = 'fp32'
    visual_encoder = create_eva_vit_g(img_size, drop_path_rate,
        use_grad_checkpoint, precision)
    ln_vision = LayerNorm(visual_encoder.num_features)
    if freeze:
        for name, param in visual_encoder.named_parameters():
            param.requires_grad = False
        visual_encoder = visual_encoder.eval()
        visual_encoder.train = disabled_train
        for name, param in ln_vision.named_parameters():
            param.requires_grad = False
        ln_vision = ln_vision.eval()
        ln_vision.train = disabled_train
        logging.info('freeze vision encoder')
    logging.info('Loading VIT Done')
    return visual_encoder, ln_vision

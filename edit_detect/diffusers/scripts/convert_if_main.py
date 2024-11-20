def main(args):
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
    text_encoder = T5EncoderModel.from_pretrained('google/t5-v1_1-xxl')
    feature_extractor = CLIPImageProcessor.from_pretrained(
        'openai/clip-vit-large-patch14')
    safety_checker = convert_safety_checker(p_head_path=args.p_head_path,
        w_head_path=args.w_head_path)
    if (args.unet_config is not None and args.unet_checkpoint_path is not
        None and args.dump_path is not None):
        convert_stage_1_pipeline(tokenizer, text_encoder, feature_extractor,
            safety_checker, args)
    if (args.unet_checkpoint_path_stage_2 is not None and args.
        dump_path_stage_2 is not None):
        convert_super_res_pipeline(tokenizer, text_encoder,
            feature_extractor, safety_checker, args, stage=2)
    if (args.unet_checkpoint_path_stage_3 is not None and args.
        dump_path_stage_3 is not None):
        convert_super_res_pipeline(tokenizer, text_encoder,
            feature_extractor, safety_checker, args, stage=3)

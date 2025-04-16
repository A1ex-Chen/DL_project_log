def convert_stage_1_pipeline(tokenizer, text_encoder, feature_extractor,
    safety_checker, args):
    unet = get_stage_1_unet(args.unet_config, args.unet_checkpoint_path)
    scheduler = DDPMScheduler(variance_type='learned_range', beta_schedule=
        'squaredcos_cap_v2', prediction_type='epsilon', thresholding=True,
        dynamic_thresholding_ratio=0.95, sample_max_value=1.5)
    pipe = IFPipeline(tokenizer=tokenizer, text_encoder=text_encoder, unet=
        unet, scheduler=scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor, requires_safety_checker=True)
    pipe.save_pretrained(args.dump_path)

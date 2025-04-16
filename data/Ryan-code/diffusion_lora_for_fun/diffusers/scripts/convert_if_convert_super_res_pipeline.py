def convert_super_res_pipeline(tokenizer, text_encoder, feature_extractor,
    safety_checker, args, stage):
    if stage == 2:
        unet_checkpoint_path = args.unet_checkpoint_path_stage_2
        sample_size = None
        dump_path = args.dump_path_stage_2
    elif stage == 3:
        unet_checkpoint_path = args.unet_checkpoint_path_stage_3
        sample_size = 1024
        dump_path = args.dump_path_stage_3
    else:
        assert False
    unet = get_super_res_unet(unet_checkpoint_path, verify_param_count=
        False, sample_size=sample_size)
    image_noising_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2')
    scheduler = DDPMScheduler(variance_type='learned_range', beta_schedule=
        'squaredcos_cap_v2', prediction_type='epsilon', thresholding=True,
        dynamic_thresholding_ratio=0.95, sample_max_value=1.0)
    pipe = IFSuperResolutionPipeline(tokenizer=tokenizer, text_encoder=
        text_encoder, unet=unet, scheduler=scheduler,
        image_noising_scheduler=image_noising_scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor,
        requires_safety_checker=True)
    pipe.save_pretrained(dump_path)

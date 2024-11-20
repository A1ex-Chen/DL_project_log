def movq(*, args, checkpoint_map_location):
    print('loading movq')
    movq_checkpoint = torch.load(args.movq_checkpoint_path, map_location=
        checkpoint_map_location)
    movq_model = movq_model_from_original_config()
    movq_diffusers_checkpoint = (
        movq_original_checkpoint_to_diffusers_checkpoint(movq_model,
        movq_checkpoint))
    del movq_checkpoint
    load_checkpoint_to_model(movq_diffusers_checkpoint, movq_model, strict=True
        )
    print('done loading movq')
    return movq_model

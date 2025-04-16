def renderer(*, args, checkpoint_map_location):
    print(' loading renderer')
    renderer_checkpoint = torch.load(args.transmitter_checkpoint_path,
        map_location=checkpoint_map_location)
    renderer_model = renderer_model_from_original_config()
    renderer_diffusers_checkpoint = (
        renderer_model_original_checkpoint_to_diffusers_checkpoint(
        renderer_model, renderer_checkpoint))
    del renderer_checkpoint
    load_checkpoint_to_model(renderer_diffusers_checkpoint, renderer_model,
        strict=True)
    print('done loading renderer')
    return renderer_model

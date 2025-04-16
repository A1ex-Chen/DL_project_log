def decoder(*, args, checkpoint_map_location):
    print('loading decoder')
    decoder_checkpoint = torch.load(args.decoder_checkpoint_path,
        map_location=checkpoint_map_location)
    decoder_checkpoint = decoder_checkpoint['state_dict']
    decoder_model = decoder_model_from_original_config()
    decoder_diffusers_checkpoint = (
        decoder_original_checkpoint_to_diffusers_checkpoint(decoder_model,
        decoder_checkpoint))
    text_proj_model = text_proj_from_original_config()
    text_proj_checkpoint = (
        text_proj_original_checkpoint_to_diffusers_checkpoint(
        decoder_checkpoint))
    load_checkpoint_to_model(text_proj_checkpoint, text_proj_model, strict=True
        )
    del decoder_checkpoint
    load_checkpoint_to_model(decoder_diffusers_checkpoint, decoder_model,
        strict=True)
    print('done loading decoder')
    return decoder_model, text_proj_model

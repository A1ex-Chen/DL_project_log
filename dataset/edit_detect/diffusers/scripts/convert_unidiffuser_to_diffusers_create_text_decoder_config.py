def create_text_decoder_config(config_type):
    if args.config_type == 'test':
        text_decoder_config = create_text_decoder_config_test()
    elif args.config_type == 'big':
        text_decoder_config = create_text_decoder_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types 'test' and 'big' are available."
            )
    return text_decoder_config

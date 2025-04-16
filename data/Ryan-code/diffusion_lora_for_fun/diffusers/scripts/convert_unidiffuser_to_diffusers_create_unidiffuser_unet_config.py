def create_unidiffuser_unet_config(config_type, version):
    if args.config_type == 'test':
        unet_config = create_unidiffuser_unet_config_test()
    elif args.config_type == 'big':
        unet_config = create_unidiffuser_unet_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types 'test' and 'big' are available."
            )
    if version == 1:
        unet_config['use_data_type_embedding'] = True
    return unet_config

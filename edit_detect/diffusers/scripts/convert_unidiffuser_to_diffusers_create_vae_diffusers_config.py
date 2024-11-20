def create_vae_diffusers_config(config_type):
    if args.config_type == 'test':
        vae_config = create_vae_diffusers_config_test()
    elif args.config_type == 'big':
        vae_config = create_vae_diffusers_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types 'test' and 'big' are available."
            )
    return vae_config

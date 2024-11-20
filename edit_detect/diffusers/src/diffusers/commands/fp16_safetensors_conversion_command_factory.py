def conversion_command_factory(args: Namespace):
    if args.use_auth_token:
        warnings.warn(
            'The `--use_auth_token` flag is deprecated and will be removed in a future version. Authentication is now handled automatically if user is logged in.'
            )
    return FP16SafetensorsCommand(args.ckpt_id, args.fp16, args.use_safetensors
        )

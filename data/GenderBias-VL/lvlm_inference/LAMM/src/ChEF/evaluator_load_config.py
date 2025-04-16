def load_config():
    args = parse_args()
    model_cfg = load_yaml(args.model_cfg)
    recipe_cfg = load_yaml(args.recipe_cfg)
    if args.batch_size is not None:
        recipe_cfg['eval_cfg']['inferencer_cfg']['batch_size'
            ] = args.batch_size
        print(f'Set batch_size to {args.batch_size}')
    save_dir = args.save_dir
    if args.debug:
        sample_len = 32
    elif args.sample_len != -1:
        sample_len = args.sample_len
    else:
        sample_len = -1
    return model_cfg, recipe_cfg, save_dir, sample_len, args.time

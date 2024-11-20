def check_and_init(args):
    """check config files and device."""
    master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1
    if args.resume:
        checkpoint_path = args.resume if isinstance(args.resume, str
            ) else find_latest_checkpoint()
        assert os.path.isfile(checkpoint_path
            ), f'the checkpoint path is not exist: {checkpoint_path}'
        LOGGER.info(
            f'Resume training from the checkpoint file :{checkpoint_path}')
        resume_opt_file_path = Path(checkpoint_path
            ).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                args = argparse.Namespace(**yaml.safe_load(f))
        else:
            LOGGER.warning(
                f"We can not find the path of {Path(checkpoint_path).parent.parent / 'args.yaml'}, we will save exp log to {Path(checkpoint_path).parent.parent}"
                )
            LOGGER.warning(
                f'In this case, make sure to provide configuration, such as data, batch size.'
                )
            args.save_dir = str(Path(checkpoint_path).parent.parent)
        args.resume = checkpoint_path
    else:
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.
            name)))
        if master_process:
            os.makedirs(args.save_dir)
    if args.specific_shape:
        if args.rect:
            LOGGER.warning(
                'You set specific shape, and rect to True is needless. YOLOv6 will use the specific shape to train.'
                )
        args.height = check_img_size(args.height, 32, floor=256)
        args.width = check_img_size(args.width, 32, floor=256)
    else:
        args.img_size = check_img_size(args.img_size, 32, floor=256)
    cfg = Config.fromfile(args.conf_file)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')
    device = select_device(args.device)
    set_random_seed(1 + args.rank, deterministic=args.rank == -1)
    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))
    return cfg, device, args

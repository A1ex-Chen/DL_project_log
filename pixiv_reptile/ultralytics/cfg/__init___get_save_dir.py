def get_save_dir(args, name=None):
    """Returns the directory path for saving outputs, derived from arguments or default settings."""
    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path
        project = args.project or (ROOT.parent / 'tests/tmp/runs' if
            TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.
            exist_ok if RANK in {-1, 0} else True)
    return Path(save_dir)

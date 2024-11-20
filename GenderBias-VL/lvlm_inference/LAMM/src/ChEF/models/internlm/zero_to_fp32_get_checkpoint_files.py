def get_checkpoint_files(checkpoint_dir, glob_pattern):
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern
        )), key=natural_keys)
    if len(ckpt_files) == 0:
        raise FileNotFoundError(
            f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")
    return ckpt_files

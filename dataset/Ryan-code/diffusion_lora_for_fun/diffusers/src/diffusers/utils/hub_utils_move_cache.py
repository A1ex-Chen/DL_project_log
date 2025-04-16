def move_cache(old_cache_dir: Optional[str]=None, new_cache_dir: Optional[
    str]=None) ->None:
    if new_cache_dir is None:
        new_cache_dir = HF_HUB_CACHE
    if old_cache_dir is None:
        old_cache_dir = old_diffusers_cache
    old_cache_dir = Path(old_cache_dir).expanduser()
    new_cache_dir = Path(new_cache_dir).expanduser()
    for old_blob_path in old_cache_dir.glob('**/blobs/*'):
        if old_blob_path.is_file() and not old_blob_path.is_symlink():
            new_blob_path = new_cache_dir / old_blob_path.relative_to(
                old_cache_dir)
            new_blob_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_blob_path, new_blob_path)
            try:
                os.symlink(new_blob_path, old_blob_path)
            except OSError:
                logger.warning(
                    'Could not create symlink between old cache and new cache. If you use an older version of diffusers again, files will be re-downloaded.'
                    )

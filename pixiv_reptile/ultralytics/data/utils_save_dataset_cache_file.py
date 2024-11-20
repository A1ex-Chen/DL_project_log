def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()
        np.save(str(path), x)
        path.with_suffix('.cache.npy').rename(path)
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(
            f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.'
            )

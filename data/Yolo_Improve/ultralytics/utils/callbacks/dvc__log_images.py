def _log_images(path, prefix=''):
    """Logs images at specified path with an optional prefix using DVCLive."""
    if live:
        name = path.name
        if (m := re.search('_batch(\\d+)', name)):
            ni = m[1]
            new_stem = re.sub('_batch(\\d+)', '_batch', path.stem)
            name = (Path(new_stem) / ni).with_suffix(path.suffix)
        live.log_image(os.path.join(prefix, name), path)

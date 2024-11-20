def _resolve_path(self, index_path, filename):
    assert os.path.isdir(index_path) or is_remote_url(index_path
        ), 'Please specify a valid ``index_path``.'
    archive_file = os.path.join(index_path, filename)
    try:
        resolved_archive_file = cached_path(archive_file)
    except EnvironmentError:
        msg = f"""Can't load '{archive_file}'. Make sure that:

- '{index_path}' is a correct remote path to a directory containing a file named {filename}- or '{index_path}' is the correct path to a directory containing a file named {filename}.

"""
        raise EnvironmentError(msg)
    if resolved_archive_file == archive_file:
        logger.info('loading file {}'.format(archive_file))
    else:
        logger.info('loading file {} from cache at {}'.format(archive_file,
            resolved_archive_file))
    return resolved_archive_file

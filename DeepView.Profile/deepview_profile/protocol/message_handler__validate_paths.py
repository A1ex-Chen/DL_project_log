def _validate_paths(project_root, entry_point):
    if not os.path.isabs(project_root):
        logger.error(
            'The project root that DeepView received is not an absolute path. This is an unexpected error. Please report a bug.'
            )
        logger.error('Current project root: %s', project_root)
        return False
    if os.path.isabs(entry_point):
        logger.error(
            'The entry point must be specified as a relative path to the current directory. Please double check that the entry point you are providing is a relative path.'
            )
        logger.error('Current entry point path: %s', entry_point)
        return False
    full_path = os.path.join(project_root, entry_point)
    if not os.path.isfile(full_path):
        logger.error(
            'Either the specified entry point is not a file or its path was specified incorrectly. Please double check that it exists and that its path is correct.'
            )
        logger.error('Current absolute path to entry point: %s', full_path)
        return False
    return True

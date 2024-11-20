def is_git_dir():
    """
    Determines whether the current file is part of a git repository. If the current file is not part of a git
    repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return GIT_DIR is not None

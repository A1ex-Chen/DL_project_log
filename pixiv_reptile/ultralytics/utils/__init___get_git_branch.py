def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name or None if not a git directory.
    """
    if IS_GIT_DIR:
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(['git', 'rev-parse',
                '--abbrev-ref', 'HEAD'])
            return origin.decode().strip()

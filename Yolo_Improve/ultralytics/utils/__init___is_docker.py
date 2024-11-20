def is_docker() ->bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    with contextlib.suppress(Exception):
        with open('/proc/self/cgroup') as f:
            return 'docker' in f.read()
    return False

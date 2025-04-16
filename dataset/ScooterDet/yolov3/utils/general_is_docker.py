def is_docker() ->bool:
    """Check if the process runs inside a docker container."""
    if Path('/.dockerenv').exists():
        return True
    try:
        with open('/proc/self/cgroup') as file:
            return any('docker' in line for line in file)
    except OSError:
        return False

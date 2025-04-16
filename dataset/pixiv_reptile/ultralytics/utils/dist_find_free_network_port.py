def find_free_network_port() ->int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def is_online() ->bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    import socket
    for host in ('1.1.1.1', '8.8.8.8', '223.5.5.5'):
        try:
            test_connection = socket.create_connection(address=(host, 53),
                timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            test_connection.close()
            return True
    return False

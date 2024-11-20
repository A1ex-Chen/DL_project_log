def is_online() ->bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    with contextlib.suppress(Exception):
        assert str(os.getenv('YOLO_OFFLINE', '')).lower() != 'true'
        import socket
        for dns in ('1.1.1.1', '8.8.8.8'):
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    return False

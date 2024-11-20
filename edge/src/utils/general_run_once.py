def run_once():
    try:
        socket.create_connection(('1.1.1.1', 443), 5)
        return True
    except OSError:
        return False

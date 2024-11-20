def default_logdir() ->str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return os.path.join('runs', current_time + '_' + socket.gethostname())

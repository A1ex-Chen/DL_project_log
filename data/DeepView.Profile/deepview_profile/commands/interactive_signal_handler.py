def signal_handler(signal, frame):
    should_shutdown.set()

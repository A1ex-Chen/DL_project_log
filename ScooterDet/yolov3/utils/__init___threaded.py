def threaded(func):

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs,
            daemon=True)
        thread.start()
        return thread
    return wrapper

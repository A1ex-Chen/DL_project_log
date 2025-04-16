def retriable_error(exception):
    return isinstance(exception, Exception)

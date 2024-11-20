def retriable_error(exception):
    return isinstance(exception, (AssertionError, redis.ConnectionError))

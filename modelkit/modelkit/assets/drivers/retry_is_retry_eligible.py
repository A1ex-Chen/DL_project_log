def is_retry_eligible(error):
    return isinstance(error, type_error) or isinstance(error, requests.
        exceptions.ChunkedEncodingError)

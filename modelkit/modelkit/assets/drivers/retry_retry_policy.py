def retry_policy(type_error=None):
    if not type_error:

        def is_retry_eligible(error):
            return isinstance(error, requests.exceptions.ChunkedEncodingError)
    else:

        def is_retry_eligible(error):
            return isinstance(error, type_error) or isinstance(error,
                requests.exceptions.ChunkedEncodingError)
    return {'wait': wait_random_exponential(multiplier=1, min=4, max=10),
        'stop': stop_after_attempt(5), 'retry': retry_if_exception(
        is_retry_eligible), 'after': log_after_retry, 'reraise': True}

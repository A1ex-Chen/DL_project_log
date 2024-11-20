def is_flaky(max_attempts: int=5, wait_before_retry: Optional[float]=None,
    description: Optional[str]=None):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):

        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1
            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)
                except Exception as err:
                    print(
                        f'Test failed with {err} at try {retry_count}/{max_attempts}.'
                        , file=sys.stderr)
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1
            return test_func_ref(*args, **kwargs)
        return wrapper
    return decorator

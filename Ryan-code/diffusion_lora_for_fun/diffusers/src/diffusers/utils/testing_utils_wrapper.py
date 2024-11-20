@functools.wraps(test_func_ref)
def wrapper(*args, **kwargs):
    retry_count = 1
    while retry_count < max_attempts:
        try:
            return test_func_ref(*args, **kwargs)
        except Exception as err:
            print(
                f'Test failed with {err} at try {retry_count}/{max_attempts}.',
                file=sys.stderr)
            if wait_before_retry is not None:
                time.sleep(wait_before_retry)
            retry_count += 1
    return test_func_ref(*args, **kwargs)

@retry(**SHORT_RETRY_POLICY)
def some_function():
    nonlocal k
    k += 1
    raise ObjectDoesNotExistError('driver', 'bucket', 'object')

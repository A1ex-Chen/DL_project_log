@wraps(func)
def wrapper(url, *args, **kwargs):
    try:
        return func(url, *args, **kwargs)
    except ClientError as exc:
        if int(exc.response['Error']['Code']) == 404:
            raise EnvironmentError('file {} not found'.format(url))
        else:
            raise

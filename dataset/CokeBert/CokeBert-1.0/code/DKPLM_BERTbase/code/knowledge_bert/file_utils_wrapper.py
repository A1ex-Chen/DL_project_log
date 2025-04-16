@wraps(func)
def wrapper(url: str, *args, **kwargs):
    try:
        return func(url, *args, **kwargs)
    except ClientError as exc:
        if int(exc.response['Error']['Code']) == 404:
            raise FileNotFoundError('file {} not found'.format(url))
        else:
            raise

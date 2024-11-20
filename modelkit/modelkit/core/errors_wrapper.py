@functools.wraps(func)
def wrapper(*args, **kwargs):
    if kwargs.pop('__internal', False):
        yield from func(*args, **kwargs)
    else:
        try:
            yield from func(*args, **kwargs)
        except PredictionError as exc:
            if os.environ.get('MODELKIT_ENABLE_SIMPLE_TRACEBACK', 'True'
                ) == 'True':
                raise strip_modelkit_traceback_frames(exc.exc) from exc
            raise exc.exc from exc
        except BaseException:
            raise

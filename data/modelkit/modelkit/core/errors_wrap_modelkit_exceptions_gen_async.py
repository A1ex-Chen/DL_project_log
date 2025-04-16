def wrap_modelkit_exceptions_gen_async(func: T) ->T:

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if kwargs.pop('__internal', False):
            async for x in func(*args, **kwargs):
                yield x
        else:
            try:
                async for x in func(*args, **kwargs):
                    yield x
            except PredictionError as exc:
                if os.environ.get('MODELKIT_ENABLE_SIMPLE_TRACEBACK', 'True'
                    ) == 'True':
                    raise strip_modelkit_traceback_frames(exc.exc) from exc
                raise exc.exc from exc
            except BaseException:
                raise
    return cast(T, wrapper)

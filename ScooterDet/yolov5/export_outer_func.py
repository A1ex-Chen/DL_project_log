def outer_func(*args, **kwargs):
    prefix = inner_args['prefix']
    try:
        with Profile() as dt:
            f, model = inner_func(*args, **kwargs)
        LOGGER.info(
            f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)'
            )
        return f, model
    except Exception as e:
        LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
        return None, None

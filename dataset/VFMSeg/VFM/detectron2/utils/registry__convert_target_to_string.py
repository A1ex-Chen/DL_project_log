def _convert_target_to_string(t: Any) ->str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__
    module_parts = module.split('.')
    for k in range(1, len(module_parts)):
        prefix = '.'.join(module_parts[:k])
        candidate = f'{prefix}.{qualname}'
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f'{module}.{qualname}'

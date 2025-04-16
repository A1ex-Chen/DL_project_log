def locate(name: str) ->Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)
    if obj is None:
        try:
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f'Cannot dynamically locate object {name}!'
                ) from e
        else:
            obj = _locate(name)
    return obj

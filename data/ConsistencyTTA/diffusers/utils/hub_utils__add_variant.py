def _add_variant(weights_name: str, variant: Optional[str]=None) ->str:
    if variant is not None:
        splits = weights_name.split('.')
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = '.'.join(splits)
    return weights_name

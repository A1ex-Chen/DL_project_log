@classmethod
def _get_signature_types(cls):
    signature_types = {}
    for k, v in inspect.signature(cls.__init__).parameters.items():
        if inspect.isclass(v.annotation):
            signature_types[k] = v.annotation,
        elif get_origin(v.annotation) == Union:
            signature_types[k] = get_args(v.annotation)
        else:
            logger.warning(
                f'cannot get type annotation for Parameter {k} of {cls}.')
    return signature_types

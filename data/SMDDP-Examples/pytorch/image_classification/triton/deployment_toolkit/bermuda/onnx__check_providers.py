def _check_providers(providers):
    providers = providers or []
    if not isinstance(providers, (list, tuple)):
        providers = [providers]
    available_providers = onnxruntime.get_available_providers()
    unavailable = set(providers) - set(available_providers)
    if unavailable:
        raise RuntimeError(f'Unavailable providers {unavailable}')
    return providers

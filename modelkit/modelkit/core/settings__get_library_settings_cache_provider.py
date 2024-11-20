def _get_library_settings_cache_provider(v: Optional[str]) ->str:
    if v is None:
        return 'none'
    elif isinstance(v, dict):
        return v.get('cache_provider', 'none')
    return getattr(v, 'cache_provider', 'none')

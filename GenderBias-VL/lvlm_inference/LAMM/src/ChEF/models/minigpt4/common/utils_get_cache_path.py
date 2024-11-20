def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path('cache_root'),
        rel_path))

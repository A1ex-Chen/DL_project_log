def clean_env():
    for env_var in ['MODELKIT_ASSETS_DIR', 'MODELKIT_ASSETS_TIMEOUT_S',
        'MODELKIT_CACHE_HOST', 'MODELKIT_CACHE_IMPLEMENTATION',
        'MODELKIT_CACHE_MAX_SIZE', 'MODELKIT_CACHE_PORT',
        'MODELKIT_CACHE_PROVIDER', 'MODELKIT_DEFAULT_PACKAGE',
        'MODELKIT_LAZY_LOADING', 'MODELKIT_STORAGE_BUCKET',
        'MODELKIT_STORAGE_FORCE_DOWNLOAD', 'MODELKIT_ASSETS_DIR_OVERRIDE',
        'MODELKIT_ASSETS_VERSIONING_SYSTEM', 'MODELKIT_STORAGE_PREFIX',
        'MODELKIT_STORAGE_PROVIDER', 'MODELKIT_STORAGE_TIMEOUT_S',
        'MODELKIT_TF_SERVING_ATTEMPTS', 'MODELKIT_TF_SERVING_ENABLE',
        'MODELKIT_TF_SERVING_MODE', 'MODELKIT_TF_SERVING_PORT']:
        os.environ.pop(env_var, None)

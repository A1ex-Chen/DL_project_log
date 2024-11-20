def __init__(self, settings: Optional[Union[Dict, LibrarySettings]]=None,
    assetsmanager_settings: Optional[dict]=None, configuration: Optional[
    Dict[str, Union[Dict[str, Any], ModelConfiguration]]]=None, models:
    Optional[LibraryModelsType]=None, required_models: Optional[Union[List[
    str], Dict[str, Any]]]=None):
    """
        Create a model library

        :param models: a `Model` class, a module, or a list of either in which the
        ModelLibrary will look for configurations.
        :param configuration: used to override configurations obtained from `models`
        :param required_models: used to restrict the models that are preloaded.
        :type settings: dict of additional settings (lazy_loading, etc.)
        :param assetsmanager_settings: settings passed to the AssetsManager
        """
    if isinstance(settings, dict):
        settings = LibrarySettings(**settings)
    self.settings: LibrarySettings = settings or LibrarySettings()
    self.assetsmanager_settings: Dict[str, Any] = assetsmanager_settings or {}
    self._override_assets_manager: Optional[AssetsManager] = None
    self._lazy_loading: bool = self.settings.lazy_loading
    if models is None:
        models = os.environ.get('MODELKIT_DEFAULT_PACKAGE')
    self.configuration: Dict[str, ModelConfiguration] = configure(models=
        models, configuration=configuration)
    self.models: Dict[str, Asset] = {}
    self.assets_info: Dict[str, AssetInfo] = {}
    self._assets_manager: Optional[AssetsManager] = None
    required_models = required_models if required_models is not None else {r:
        {} for r in self.configuration}
    if isinstance(required_models, list):
        required_models = {r: {} for r in required_models}
    self.required_models: Dict[str, Dict[str, Any]] = required_models
    self.cache: Optional[Cache] = None
    if self.settings.cache:
        if isinstance(self.settings.cache, RedisSettings):
            try:
                self.cache = RedisCache(self.settings.cache.host, self.
                    settings.cache.port)
            except (ConnectionError, redis.ConnectionError) as e:
                logger.error('Cannot ping redis instance', cache_host=self.
                    settings.cache.host, port=self.settings.cache.port)
                raise RedisCacheException(
                    f'Cannot ping redis instance[cache_host={self.settings.cache.host}, port={self.settings.cache.port}]'
                    ) from e
        if isinstance(self.settings.cache, NativeCacheSettings):
            self.cache = NativeCache(self.settings.cache.implementation,
                self.settings.cache.maxsize)
    if not self._lazy_loading:
        self.preload()

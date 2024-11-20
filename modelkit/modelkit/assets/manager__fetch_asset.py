def _fetch_asset(self, spec: AssetSpec, _force_download=False):
    with ContextualizedLogging(name=spec.name):
        self._resolve_version(spec)
        with ContextualizedLogging(version=spec.version):
            logger.debug('Resolved latest version', version=spec.version)
            return self._fetch_asset_version(spec, _force_download)

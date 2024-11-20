def _resolve_version(self, spec: AssetSpec) ->None:
    local_versions = self._list_local_versions(spec)
    logger.debug('Local versions', local_versions=local_versions)
    if spec.is_version_complete():
        return
    remote_versions = []
    if self.storage_provider:
        remote_versions = self.storage_provider.get_versions_info(spec.name)
        logger.debug('Fetched remote versions', remote_versions=remote_versions
            )
    all_versions = spec.sort_versions(version_list=set(local_versions +
        remote_versions))
    if not all_versions:
        if not spec.version:
            logger.debug('Asset has no version information')
            return None
        raise errors.LocalAssetDoesNotExistError(name=spec.name, version=
            spec.version, local_versions=local_versions)
    spec.set_latest_version(all_versions)

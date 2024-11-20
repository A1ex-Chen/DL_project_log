def _fetch_asset_version(self, spec: AssetSpec, _force_download: bool) ->Dict[
    str, Any]:
    local_path = os.path.join(self.assets_dir, *spec.name.split('/'), spec.
        version or '')
    if not spec.version:
        return _fetch_local_version(spec.name, os.path.join(self.assets_dir,
            *spec.name.split('/')))
    if not self.storage_provider:
        if _force_download:
            raise errors.StorageDriverError(
                'can not force_download with no storage provider')
        local_versions = self._list_local_versions(spec)
        if spec.version not in local_versions:
            raise errors.LocalAssetDoesNotExistError(name=spec.name,
                version=spec.version, local_versions=local_versions)
        asset_dict = {'from_cache': True, 'version': spec.version, 'path':
            local_path}
    else:
        lock_path = os.path.join(self.assets_dir, '.cache', *spec.name.
            split('/')) + '.lock'
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with filelock.FileLock(lock_path, timeout=self.timeout):
            local_versions = self._list_local_versions(spec)
            if not _has_succeeded(local_path):
                logger.info(
                    'Previous fetching of asset has failed, redownloading.')
                _force_download = True
            if not _force_download and spec.version in local_versions:
                asset_dict = {'from_cache': True, 'version': spec.version,
                    'path': local_path}
            else:
                if _force_download:
                    if os.path.exists(local_path):
                        if os.path.isdir(local_path):
                            shutil.rmtree(local_path)
                        else:
                            os.unlink(local_path)
                    success_object_path = _success_file_path(local_path)
                    if os.path.exists(success_object_path):
                        os.unlink(success_object_path)
                logger.info('Fetching distant asset', local_versions=
                    local_versions)
                asset_download_info = self.storage_provider.download(spec.
                    name, spec.version, self.assets_dir)
                asset_dict = {**asset_download_info, 'from_cache': False,
                    'version': spec.version, 'path': local_path}
                open(_success_file_path(local_path), 'w').close()
    if spec.sub_part:
        local_sub_part = os.path.join(*(list(os.path.split(str(asset_dict[
            'path']))) + [p for p in spec.sub_part.split('/') if p]))
        asset_dict['path'] = local_sub_part
    return asset_dict

def update(self, asset_path: str, name: str, version: str, dry_run=False):
    """
        Update an existing asset version
        """
    spec = AssetSpec(name=name, version=version)
    versions_object_name = self.get_versions_object_name(spec.name)
    if not self.driver.exists(versions_object_name):
        raise errors.AssetDoesNotExistError(spec.name)
    logger.info('Updating asset', name=spec.name, version=spec.version,
        asset_path=asset_path)
    versions_list = self.get_versions_info(spec.name)
    self.push(asset_path, spec.name, spec.version, dry_run=dry_run)
    with tempfile.TemporaryDirectory() as tmp_dir:
        versions_fn = os.path.join(tmp_dir, 'versions.json')
        versions = spec.sort_versions([spec.version] + versions_list)
        with open(versions_fn, 'w') as f:
            json.dump({'versions': versions}, f)
        logger.debug('Pushing updated versions file', name=spec.name,
            versions=versions)
        if not dry_run:
            self.driver.upload_object(versions_fn, versions_object_name)

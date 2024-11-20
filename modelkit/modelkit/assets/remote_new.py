def new(self, asset_path: str, name: str, version: str, dry_run=False):
    """
        Upload a new asset
        """
    versions_object_name = self.get_versions_object_name(name)
    if self.driver.exists(versions_object_name):
        raise errors.AssetAlreadyExistsError(name)
    logger.info('Pushing new asset', name=name, asset_path=asset_path)
    self.push(asset_path, name, version, dry_run=dry_run)
    with tempfile.TemporaryDirectory() as dversions:
        with open(os.path.join(dversions, 'versions.json'), 'w') as f:
            json.dump({'versions': [version]}, f)
        logger.debug('Pushing versions file', name=name)
        if not dry_run:
            self.driver.upload_object(os.path.join(dversions,
                'versions.json'), versions_object_name)

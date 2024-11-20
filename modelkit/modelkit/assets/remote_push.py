def push(self, asset_path, name, version, dry_run=False):
    """
        Push asset
        """
    with ContextualizedLogging(name=name, version=version, asset_path=
        asset_path):
        logger.info('Pushing asset')
        object_name = self.get_object_name(name, version)
        if self.driver.exists(object_name):
            raise errors.AssetAlreadyExistsError(
                f'`{name}` already exists, cannot overwrite asset for version `{version}`'
                )
        meta = {'push_date': datetime.datetime.now(tz.UTC).isoformat(),
            'is_directory': os.path.isdir(asset_path)}
        if meta['is_directory']:
            asset_path += '/' if not asset_path.endswith('/') else ''
            meta['contents'] = sorted(f[len(asset_path):] for f in glob.
                iglob(os.path.join(asset_path, '**/*'), recursive=True) if
                os.path.isfile(f))
            logger.info('Pushing multi-part asset file', n_parts=len(meta[
                'contents']))
            for part_no, part in enumerate(meta['contents']):
                path_to_push = os.path.join(asset_path, part)
                remote_object_name = '/'.join(x for x in object_name.split(
                    '/') + list(os.path.split(part)) if x)
                logger.debug('Pushing multi-part asset file', object_name=
                    remote_object_name, path_to_push=path_to_push, part=
                    part, part_no=part_no, n_parts=len(meta['contents']))
                if not dry_run:
                    self.driver.upload_object(path_to_push, remote_object_name)
            logger.info('Pushed multi-part asset file', n_parts=len(meta[
                'contents']))
        else:
            logger.info('Pushing asset file', object_name=object_name)
            if not dry_run:
                self.driver.upload_object(asset_path, object_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_file_path = os.path.join(tmp_dir, 'asset.meta')
            with open(meta_file_path, 'w', encoding='utf-8') as fmeta:
                json.dump(meta, fmeta)
            logger.debug('Pushing meta file', meta=meta, meta_object_name=
                object_name + '.meta')
            if not dry_run:
                self.driver.upload_object(meta_file_path, object_name + '.meta'
                    )

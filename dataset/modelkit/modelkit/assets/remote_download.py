def download(self, name, version, destination):
    """
        Retrieves the asset and returns a dictionary with meta information, asset
        origin (from cache) and local path
        """
    with ContextualizedLogging(name=name, version=version):
        destination_path = os.path.join(destination, *name.split('/'), version)
        object_name = self.get_object_name(name, version)
        meta = self.get_asset_meta(name, version)
        if meta.get('is_directory'):
            logger.info('Downloading remote multi-part asset', n_parts=len(
                meta['contents']))
            t0 = time.monotonic()
            for part_no, part in enumerate(meta['contents']):
                current_destination_path = os.path.join(destination_path, *
                    part.split('/'))
                os.makedirs(os.path.dirname(current_destination_path),
                    exist_ok=True)
                remote_part_name = '/'.join(x for x in object_name.split(
                    '/') + part.split('/') if x)
                logger.debug('Downloading asset part', part_no=part_no,
                    n_parts=len(meta['contents']))
                self.driver.download_object(remote_part_name,
                    current_destination_path)
                size = get_size(current_destination_path)
                logger.debug('Downloaded asset part', part_no=part_no,
                    n_parts=len(meta['contents']), size=humanize.
                    naturalsize(size), size_bytes=size)
            size = get_size(destination_path)
            logger.info('Downloaded remote multi-part asset', size=humanize
                .naturalsize(size), size_bytes=size)
        else:
            logger.info('Downloading remote asset')
            t0 = time.monotonic()
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            self.driver.download_object(object_name, destination_path)
            size = get_size(destination_path)
            download_time = time.monotonic() - t0
            logger.info('Downloaded asset', size=humanize.naturalsize(size),
                time=humanize.naturaldelta(datetime.timedelta(seconds=
                download_time)), time_seconds=download_time, size_bytes=size)
        return {'path': destination_path, 'meta': meta}

def _download_ann(self):
    """
        Download annotation files if necessary.
        All the vision-language datasets should have annotations of unified format.

        storage_path can be:
          (1) relative/absolute: will be prefixed with env.cache_root to make full path if relative.
          (2) basename/dirname: will be suffixed with base name of URL if dirname is provided.

        Local annotation paths should be relative.
        """
    anns = self.config.build_info.annotations
    splits = anns.keys()
    cache_root = registry.get_path('cache_root')
    for split in splits:
        info = anns[split]
        urls, storage_paths = info.get('url', None), info.storage
        if isinstance(urls, str):
            urls = [urls]
        if isinstance(storage_paths, str):
            storage_paths = [storage_paths]
        assert len(urls) == len(storage_paths)
        for url_or_filename, storage_path in zip(urls, storage_paths):
            if not os.path.isabs(storage_path):
                storage_path = os.path.join(cache_root, storage_path)
            dirname = os.path.dirname(storage_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if os.path.isfile(url_or_filename):
                src, dst = url_or_filename, storage_path
                if not os.path.exists(dst):
                    shutil.copyfile(src=src, dst=dst)
                else:
                    logging.info('Using existing file {}.'.format(dst))
            else:
                if os.path.isdir(storage_path):
                    raise ValueError(
                        'Expecting storage_path to be a file path, got directory {}'
                        .format(storage_path))
                else:
                    filename = os.path.basename(storage_path)
                download_url(url=url_or_filename, root=dirname, filename=
                    filename)

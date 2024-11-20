def get_asset_meta(self, name, version):
    """
        Retrieve asset metadata
        """
    meta_object_name = self.get_meta_object_name(name, version)
    with tempfile.TemporaryDirectory() as tempdir:
        fdst = os.path.join(tempdir, 'meta.tmp')
        self.driver.download_object(meta_object_name, fdst)
        with open(fdst) as f:
            meta = json.load(f)
        meta['push_date'] = parser.isoparse(meta['push_date'])
    return meta

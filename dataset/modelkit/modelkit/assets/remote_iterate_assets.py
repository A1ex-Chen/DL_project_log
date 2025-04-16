def iterate_assets(self):
    assets_set = set()
    for asset_path in self.driver.iterate_objects(self.prefix):
        if asset_path.endswith('.versions'):
            asset_name = '/'.join(asset_path[len(self.prefix) + 1:-len(
                '.versions')].split('/'))
            assets_set.add(asset_name)
    for asset_name in sorted(assets_set):
        versions_list = self.get_versions_info(asset_name)
        yield asset_name, versions_list

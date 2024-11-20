def _list_local_versions(self, spec: AssetSpec) ->List[str]:
    local_name = os.path.join(self.assets_dir, *spec.name.split('/'))
    return spec.get_local_versions(local_name)

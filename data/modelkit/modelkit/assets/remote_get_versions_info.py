def get_versions_info(self, name):
    """
        Retrieve asset versions information
        """
    versions_object_name = self.get_versions_object_name(name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        versions_object_path = os.path.join(tmp_dir, name + '.version')
        os.makedirs(os.path.dirname(versions_object_path), exist_ok=True)
        self.driver.download_object(versions_object_name, versions_object_path)
        with open(versions_object_path) as f:
            versions_list = json.load(f)['versions']
        return versions_list

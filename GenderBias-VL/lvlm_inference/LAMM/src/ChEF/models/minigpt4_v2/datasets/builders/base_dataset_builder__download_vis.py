def _download_vis(self):
    storage_path = self.config.build_info.get(self.data_type).storage
    storage_path = utils.get_cache_path(storage_path)
    if not os.path.exists(storage_path):
        warnings.warn(
            f"""
                The specified path {storage_path} for visual inputs does not exist.
                Please provide a correct path to the visual inputs or
                refer to datasets/download_scripts/README.md for downloading instructions.
                """
            )

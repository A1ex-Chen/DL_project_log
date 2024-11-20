@staticmethod
def query(config_path: str) ->Optional[str]:
    """
        Args:
            config_path: relative config filename
        """
    name = config_path.replace('.yaml', '').replace('.py', '')
    if name in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[name]
        return _ModelZooUrls.S3_PREFIX + name + '/' + suffix
    return None

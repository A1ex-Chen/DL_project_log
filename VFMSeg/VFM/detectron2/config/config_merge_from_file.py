def merge_from_file(self, cfg_filename: str, allow_unsafe: bool=True) ->None:
    """
        Load content from the given config file and merge it into self.

        Args:
            cfg_filename: config filename
            allow_unsafe: allow unsafe yaml syntax
        """
    assert PathManager.isfile(cfg_filename
        ), f"Config file '{cfg_filename}' does not exist!"
    loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=
        allow_unsafe)
    loaded_cfg = type(self)(loaded_cfg)
    from .defaults import _C
    latest_ver = _C.VERSION
    assert latest_ver == self.VERSION, 'CfgNode.merge_from_file is only allowed on a config object of latest version!'
    logger = logging.getLogger(__name__)
    loaded_ver = loaded_cfg.get('VERSION', None)
    if loaded_ver is None:
        from .compat import guess_version
        loaded_ver = guess_version(loaded_cfg, cfg_filename)
    assert loaded_ver <= self.VERSION, 'Cannot merge a v{} config into a v{} config.'.format(
        loaded_ver, self.VERSION)
    if loaded_ver == self.VERSION:
        self.merge_from_other_cfg(loaded_cfg)
    else:
        from .compat import upgrade_config, downgrade_config
        logger.warning(
            "Loading an old v{} config file '{}' by automatically upgrading to v{}. See docs/CHANGELOG.md for instructions to update your files."
            .format(loaded_ver, cfg_filename, self.VERSION))
        old_self = downgrade_config(self, to_version=loaded_ver)
        old_self.merge_from_other_cfg(loaded_cfg)
        new_config = upgrade_config(old_self)
        self.clear()
        self.update(new_config)

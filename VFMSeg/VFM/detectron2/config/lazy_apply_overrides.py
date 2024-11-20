@staticmethod
def apply_overrides(cfg, overrides: List[str]):
    """
        In-place override contents of cfg.

        Args:
            cfg: an omegaconf config object
            overrides: list of strings in the format of "a=b" to override configs.
                See https://hydra.cc/docs/next/advanced/override_grammar/basic/
                for syntax.

        Returns:
            the cfg object
        """

    def safe_update(cfg, key, value):
        parts = key.split('.')
        for idx in range(1, len(parts)):
            prefix = '.'.join(parts[:idx])
            v = OmegaConf.select(cfg, prefix, default=None)
            if v is None:
                break
            if not OmegaConf.is_config(v):
                raise KeyError(
                    f'Trying to update key {key}, but {prefix} is not a config, but has type {type(v)}.'
                    )
        OmegaConf.update(cfg, key, value, merge=True)
    from hydra.core.override_parser.overrides_parser import OverridesParser
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(overrides)
    for o in overrides:
        key = o.key_or_group
        value = o.value()
        if o.is_delete():
            raise NotImplementedError(
                'deletion is not yet a supported override')
        safe_update(cfg, key, value)
    return cfg

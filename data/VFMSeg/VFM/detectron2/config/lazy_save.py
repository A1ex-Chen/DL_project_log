@staticmethod
def save(cfg, filename: str):
    """
        Save a config object to a yaml file.
        Note that when the config dictionary contains complex objects (e.g. lambda),
        it can't be saved to yaml. In that case we will print an error and
        attempt to save to a pkl file instead.

        Args:
            cfg: an omegaconf config object
            filename: yaml file name to save the config file
        """
    logger = logging.getLogger(__name__)
    try:
        cfg = deepcopy(cfg)
    except Exception:
        pass
    else:

        def _replace_type_by_name(x):
            if '_target_' in x and callable(x._target_):
                try:
                    x._target_ = _convert_target_to_string(x._target_)
                except AttributeError:
                    pass
        _visit_dict_config(cfg, _replace_type_by_name)
    save_pkl = False
    try:
        dict = OmegaConf.to_container(cfg, resolve=False)
        dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=
            True, width=9999)
        with PathManager.open(filename, 'w') as f:
            f.write(dumped)
        try:
            _ = yaml.unsafe_load(dumped)
        except Exception:
            logger.warning(
                f'The config contains objects that cannot serialize to a valid yaml. {filename} is human-readable but cannot be loaded.'
                )
            save_pkl = True
    except Exception:
        logger.exception('Unable to serialize the config to yaml. Error:')
        save_pkl = True
    if save_pkl:
        new_filename = filename + '.pkl'
        try:
            with PathManager.open(new_filename, 'wb') as f:
                cloudpickle.dump(cfg, f)
            logger.warning(
                f'Config is saved using cloudpickle at {new_filename}.')
        except Exception:
            pass

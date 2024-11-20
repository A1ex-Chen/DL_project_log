@staticmethod
def build_preprocess_config(config):
    preprocess = config.get('preprocess', None)
    assert preprocess is not None, 'Missing preprocess configuration file.'
    preprocess_config = OmegaConf.create()
    preprocess_config = OmegaConf.merge(preprocess_config, {'preprocess':
        config['preprocess']})
    return preprocess_config

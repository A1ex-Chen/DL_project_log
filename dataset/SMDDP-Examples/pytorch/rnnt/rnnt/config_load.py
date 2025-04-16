def load(fpath, max_duration=None):
    if fpath.endswith('.toml'):
        raise ValueError('.toml config format has been changed to .yaml')
    cfg = yaml.safe_load(open(fpath, 'r'))
    yaml.Dumper.ignore_aliases = lambda *args: True
    cfg = yaml.safe_load(yaml.dump(cfg))
    if max_duration is not None:
        cfg['input_train']['audio_dataset']['max_duration'] = max_duration
        cfg['input_train']['filterbank_features']['max_duration'
            ] = max_duration
    return cfg

def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
    if cfg_dict is None:
        cfg_dict = dict()
    elif not isinstance(cfg_dict, dict):
        raise TypeError('cfg_dict must be a dict, but got {}'.format(type(
            cfg_dict)))
    super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
    super(Config, self).__setattr__('_filename', filename)
    if cfg_text:
        text = cfg_text
    elif filename:
        with open(filename, 'r') as f:
            text = f.read()
    else:
        text = ''
    super(Config, self).__setattr__('_text', text)

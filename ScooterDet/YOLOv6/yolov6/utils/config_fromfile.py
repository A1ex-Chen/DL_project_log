@staticmethod
def fromfile(filename):
    cfg_dict, cfg_text = Config._file2dict(filename)
    return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

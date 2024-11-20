@staticmethod
def _file2dict(filename):
    filename = str(filename)
    if filename.endswith('.py'):
        with tempfile.TemporaryDirectory() as temp_config_dir:
            shutil.copyfile(filename, osp.join(temp_config_dir,
                '_tempconfig.py'))
            sys.path.insert(0, temp_config_dir)
            mod = import_module('_tempconfig')
            sys.path.pop(0)
            cfg_dict = {name: value for name, value in mod.__dict__.items() if
                not name.startswith('__')}
            del sys.modules['_tempconfig']
    else:
        raise IOError('Only .py type are supported now!')
    cfg_text = filename + '\n'
    with open(filename, 'r') as f:
        cfg_text += f.read()
    return cfg_dict, cfg_text

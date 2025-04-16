@staticmethod
def _load_metadata(f=Path('path/to/meta.yaml')):
    if f.exists():
        d = yaml_load(f)
        return d['stride'], d['names']
    return None, None
